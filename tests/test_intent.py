"""意图分流的阈值策略。

这个文件的重点是**策略**，不是模型 —— 所以用一个假的 LLM 把
"模型返回了什么标签 + 多少置信度"直接喂进去，看路由器怎么处理。
真实模型的不确定性留在上游，这里只锁死我们自己的判断规则。

为什么值得单独测？因为这段代码的一次真实事故：
同一条正经问题连问三次，拿到过 rag(0.60) / refuse(0.85) / rag(0.45)。
模型自报的置信度校准很差，**它给 0.85 的那个判断是错的，给 0.60 的那个是对的**。
既然这个数字不可信，那"怎么用它"就成了本项目里最需要被钉死的逻辑之一。
"""

from __future__ import annotations

import pytest

from citepilot.intent import LOW, HIGH, NEED_CERTAIN, IntentRouter


class _FakeRes:
    """冒充 LLM 的返回元信息。意图分流只用到这几个字段。"""

    ms = 1
    prompt_tokens = 1
    completion_tokens = 1


class _FakeLLM:
    """按预置的 (route, confidence) 回答，不联网、不随机。"""

    def __init__(self, route: str, confidence: float, reason: str = "fake"):
        self._payload = {"route": route, "confidence": confidence, "reason": reason}

    def complete_json(self, messages, **kwargs):  # noqa: ARG002
        return dict(self._payload), _FakeRes()


def route_of(question: str, route: str, confidence: float) -> str:
    return IntentRouter(llm=_FakeLLM(route, confidence)).classify(question).route


# 一条不会命中 L0 规则的中性问法，保证每次都会走到 LLM 那一层。
Q = "极光项目有几个哨兵节点？"


# ----------------------------------------------------------------------
# 核心不变量：refuse / clarify 是"不查资料"的出口，门槛单独抬高到 0.90
# ----------------------------------------------------------------------
def test_refuse_below_certain_is_downgraded_to_rag():
    """模型说"拒答"，但只有 0.85 的把握 —— 去查资料，别把用户拒掉。

    这就是线上真实发生过的那一次（refuse @ 0.85 把一条正常问题判成"我不能替你做"）。
    """
    assert route_of(Q, "refuse", 0.85) == "rag"


def test_clarify_below_certain_is_downgraded_to_rag():
    """同理：反问也让用户什么都拿不到，门槛一样要高。"""
    assert route_of(Q, "clarify", 0.80) == "rag"


@pytest.mark.parametrize("route", ["refuse", "clarify"])
def test_deny_routes_survive_when_confident_enough(route):
    """把握足够大时，必须尊重模型 —— 否则"拒答"这个能力就形同虚设。

    （真正该拒的问题由 L0 规则兜底，例如"帮我转账"；
      这里守的是 LLM 才能识别的那类，例如"预测明天比特币价格"。）
    """
    assert route_of(Q, route, NEED_CERTAIN) == route
    assert route_of(Q, route, 0.99) == route


def test_deny_threshold_is_strictly_higher_than_normal():
    """门槛必须真的更高，不能只是"换个名字"。

    这条断言是防回归的：将来有人把 NEED_CERTAIN 调成 0.6，
    整个设计意图（拒绝比检索贵）就没了，但功能测试全绿、没人会发现。
    """
    assert NEED_CERTAIN > HIGH > LOW


# ----------------------------------------------------------------------
# 原有档位不能被这次改动碰坏
# ----------------------------------------------------------------------
def test_mid_confidence_still_prefers_retrieval():
    """中档偏保守走检索 —— 这是原本就有的策略。"""
    assert route_of(Q, "rag", 0.60) == "rag"
    assert route_of(Q, "direct", 0.60) == "rag"


def test_low_confidence_asks_back():
    """低置信：连"这算什么"都说不清，反问是诚实的。"""
    assert route_of(Q, "rag", 0.30) == "clarify"


def test_high_confidence_rag_is_untouched():
    """rag 是"做事"的出口，门槛不抬，0.9 以上照常走。"""
    assert route_of(Q, "rag", 0.92) == "rag"


def test_invalid_route_falls_through():
    """模型吐了非法枚举值也不能崩，且不能凭这个把用户拒掉。"""
    assert route_of(Q, "nonsense", 0.99) in ("rag", "clarify")


# ----------------------------------------------------------------------
# 降级必须可观测 —— 否则事后查不出"这个问题为什么去检索了"
# ----------------------------------------------------------------------
def test_downgrade_is_recorded_in_trace():
    res = IntentRouter(llm=_FakeLLM("refuse", 0.85)).classify(Q)
    assert res.route == "rag"
    assert res.layer == "L1_llm"
    assert res.steps and res.steps[0].name == "intent_llm"
    assert res.steps[0].meta.get("downgraded_from") == "refuse"
    # 置信度还原样留着，不做美化 —— 排查时需要看到模型到底报了多少
    assert res.steps[0].meta.get("confidence") == 0.85


def test_no_downgrade_marker_when_not_downgraded():
    """没发生降级就不该留下这个字段，否则统计会算错。"""
    res = IntentRouter(llm=_FakeLLM("rag", 0.95)).classify(Q)
    assert "downgraded_from" not in res.steps[0].meta


# ----------------------------------------------------------------------
# L0 规则层必须继续拦住该拦的（不花模型调用）
# ----------------------------------------------------------------------
@pytest.mark.parametrize("q,expected", [
    ("你好", "direct"),
    ("你是谁", "direct"),
    ("帮我转账 500 块", "refuse"),
    ("", "clarify"),
])
def test_l0_rules_still_short_circuit(q, expected):
    router = IntentRouter(llm=_FakeLLM("rag", 0.99))
    res = router.classify(q)
    assert res.route == expected
    assert res.layer == "L0_rule"
    assert res.steps == []      # 命中规则就不该产生任何模型步骤
