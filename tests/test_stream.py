"""流式链路的测试。

流式代码有一类特别隐蔽的 bug：**它在"正常情况"下几乎总是对的，
只在网络分片、上游异常、或校验不通过时露馅**。
而这几条路径恰恰没法靠"手动跑一次看看"覆盖 ——
手动跑十次都不会撞上"引用校验不通过"这一支。

所以这里把三条关键路径都钉住：
  1) StreamHandle 对上游 chunk 的各种怪形状的容忍（空 choices / reasoning / usage 位置）
  2) 引用校验失败 → 发 revoke 事件 → 重写（这是流式与校验的接缝）
  3) GenStream 的降级路径（两轮都过不了 → 抽取式）

注意：这些测试**不在测试里复刻一份解析逻辑**，而是用 monkeypatch
把 litellm.completion 换成预设的 chunk 序列，让真实的 StreamHandle
去消费它。复刻一份的话，两份代码迟早会走偏 ——
而且走偏的那份永远是测试里那份，等于白测。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citepilot.answer import AnswerGenerator, GenStream
from citepilot.llm import LLM, LLMResult, StreamHandle
from citepilot.schema import Chunk, Hit


# ------------------------------------------------------------------
# 造数据的小工具
# ------------------------------------------------------------------
def _hit(i: int, text: str = "正文内容用于引用摘录") -> Hit:
    c = Chunk(
        chunk_id=f"ch-01::c{i:03d}", chapter_id="ch-01", chapter_no="第 1 章",
        chapter_title="支付的本质", part="第一篇", section_title="1.1 三流合一",
        text=text,
    )
    return Hit(chunk=c, score=1.0 / i)


def _delta(content: str | None = None, reasoning: str | None = None):
    """模拟一个正常的内容/推理 chunk。"""
    d = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=d)], usage=None)


def _usage_only(prompt: int, completion: int):
    """模拟只带 usage、choices 为空列表的收尾 chunk。

    这是真实上游会发的形状（OpenAI 与 DeepSeek 都这么发），
    而它正是"流式代码最经典的一行崩溃"来源：直接取 choices[0] 会 IndexError。
    """
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _no_delta():
    """choices 非空但 delta 为 None（部分上游在首帧会这么发）。"""
    return SimpleNamespace(choices=[SimpleNamespace(delta=None)], usage=None)


@pytest.fixture
def upstream(monkeypatch):
    """把 litellm.completion 换成"照本宣科的假上游"。

    返回一个安装器：传进去一组 chunk，真实的 StreamHandle 就会消费它们。
    """
    import litellm

    def _install(chunks, *, fail_with_usage_option: bool = False):
        calls = {"n": 0}

        def fake_completion(**kwargs):
            calls["n"] += 1
            # 模拟"provider 不认 stream_options"的场景
            if fail_with_usage_option and "stream_options" in kwargs:
                raise ValueError("unsupported parameter: stream_options")
            return iter(chunks)

        monkeypatch.setattr(litellm, "completion", fake_completion)
        return calls

    return _install


def _new_handle(**kw) -> StreamHandle:
    # 默认给一段非空 messages：token 兜底估算正是按 prompt 字符数算的，
    # 传空列表会让"估算"这条路测不到东西。
    kw.setdefault("messages",
                  [{"role": "user", "content": "这是一段用于测试 token 估算的提示词，长度足够触发非零结果"}])
    return StreamHandle(model="m", api_key="k", api_base="u",
                        temperature=0.1, max_tokens=10, timeout=5,
                        parent=LLM(model="m", api_key="k"), **kw)


# ------------------------------------------------------------------
# 1) StreamHandle 对上游 chunk 形状的容忍
# ------------------------------------------------------------------
def test_usage_chunk_with_empty_choices_does_not_crash(upstream):
    """只带 usage、choices 为空的收尾 chunk 必须被安全跳过。

    这是流式解析最容易写崩的一行：choices[0] 在空列表上直接 IndexError，
    而且它出现在流的最后 —— 前面内容都正常，最后突然断掉，
    排查时很容易误判成"上游网络问题"。
    """
    upstream([_delta("你好"), _usage_only(10, 5)])
    h = _new_handle()
    events = list(h)

    assert [e["text"] for e in events] == ["你好"]
    assert h.prompt_tokens == 10
    assert h.completion_tokens == 5


def test_chunk_with_none_delta_is_skipped(upstream):
    """delta 为 None 的帧也要能跳过（有些上游用它发心跳）。"""
    upstream([_no_delta(), _delta("内容")])
    events = list(_new_handle())
    assert [e["text"] for e in events] == ["内容"]


def test_falls_back_when_provider_rejects_stream_options(upstream):
    """provider 不认 stream_options 时要能退一步重试，而不是直接失败。

    有些上游（尤其自建/代理的 OpenAI 兼容服务）会因为多传了
    stream_options 直接 400。如果这里不兜住，流式在这些上游上完全不可用。
    """
    calls = upstream([_delta("ok")], fail_with_usage_option=True)
    events = list(_new_handle())
    assert [e["text"] for e in events] == ["ok"]
    assert calls["n"] == 2, "应当先带 stream_options 试一次，失败后再不带试一次"


def test_reasoning_is_separated_and_batched(upstream):
    """推理模型的思考片段不能混进正文，而且要被攒批发出去。

    两件事一起验：
      ① 不能混 —— 混进去用户会看到大段"让我想想…"的草稿，
         而且这些草稿还会被引用校验当成正文去查角标，双重错误。
      ② 要攒批 —— 实测逐 token 转发会产生 150+ 个只含一两个字的推理事件，
         帧头开销比内容还大。攒批后事件数降一个数量级。
    """
    chunks = []
    for ch in "让我想想，需要引用资料[1]。":
        chunks.append(_delta(reasoning=ch))
    chunks.append(_delta("备付金"))
    chunks.append(_delta("是集中缴存的[1]。"))
    upstream(chunks)

    # 门槛设大一点，确保只有"正文开始"那一下会把缓冲冲出来
    h = _new_handle(reasoning_flush_chars=1000, reasoning_flush_ms=100000)
    events = list(h)

    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    content = "".join(e["text"] for e in events if e["type"] == "content")

    assert content == "备付金是集中缴存的[1]。"
    assert reasoning == "让我想想，需要引用资料[1]。"
    assert "让我想想" not in content
    assert h.reasoning_chars == len(reasoning)

    # 攒批生效：15 个推理 token 只应产出 1 个 reasoning 事件
    assert sum(1 for e in events if e["type"] == "reasoning") == 1


def test_reasoning_is_flushed_before_content_starts(upstream):
    """正文出来之前必须先冲掉思考缓冲，否则会出现"正文在前、思考在后"的错序。"""
    upstream([_delta(reasoning="先想"), _delta("正文")])
    events = list(_new_handle(reasoning_flush_chars=1000, reasoning_flush_ms=100000))
    types = [e["type"] for e in events]
    assert types == ["reasoning", "content"], f"顺序错了：{types}"


def test_estimated_tokens_when_upstream_sends_no_usage(upstream):
    """上游全程不给 usage 时要兜底估算 —— 宁可略偏，也要让成本表有数。"""
    upstream([_delta("一二三四五"), _delta("六七八九十")])
    h = _new_handle()
    list(h)
    assert h.prompt_tokens > 0
    assert h.completion_tokens > 0


# ------------------------------------------------------------------
# 2) 引用校验与流式的接缝：revoke
# ------------------------------------------------------------------
class FakeLLM(LLM):
    """假的 LLM：流式部分交给真实的 StreamHandle（配合 upstream fixture），
    非流式部分（修复轮）直接返回预设文本。"""

    def __init__(self, repair_text: str = ""):
        super().__init__(model="fake/model", api_key="sk-fake")
        self._repair_text = repair_text
        self.complete_calls = 0

    @property
    def enabled(self) -> bool:  # type: ignore[override]
        return True

    def complete(self, messages, **kw):
        self.complete_calls += 1
        return LLMResult(text=self._repair_text, prompt_tokens=200,
                         completion_tokens=40, ms=50, model="fake")


def _drain(gs: GenStream) -> list[dict]:
    return list(gs)


def test_stream_happy_path_emits_deltas_then_verify(upstream):
    upstream([_delta("备付金是集中缴存的"), _delta("[1]。")])
    gs = AnswerGenerator(llm=FakeLLM()).generate_stream("备付金", [_hit(1)])
    events = _drain(gs)
    types = [e["type"] for e in events]

    assert types.count("delta") == 2
    assert "revoke" not in types, "引用合法时不该撤回"
    verify = next(e for e in events if e["type"] == "verify")
    assert verify["grounded"] is True
    assert gs.result is not None
    assert gs.result.answer.grounded is True
    assert [c.marker for c in gs.result.answer.citations] == [1]
    assert gs.revoked is False


def test_stream_revokes_when_citation_out_of_range(upstream):
    """模型引用了不存在的来源 → 必须发 revoke，让前端把已展示的内容清掉。

    这是"流式"与"引用校验"唯一的接缝：内容已经推出去了，
    校验才发现有问题。处理方式必须显式（发事件），不能静默替换 ——
    否则用户会看到答案自己变了，像见鬼。
    """
    upstream([_delta("这是引用幻觉"), _delta("[7]。")])
    llm = FakeLLM(repair_text="修正后使用正确来源[1]。")
    gs = AnswerGenerator(llm=llm).generate_stream("问题", [_hit(1)])
    events = _drain(gs)
    types = [e["type"] for e in events]

    assert "revoke" in types, "引用越界必须触发撤回"
    rev = next(e for e in events if e["type"] == "revoke")
    assert "7" in rev["reason"]
    assert gs.revoked is True

    # 修复轮走非流式，产出最终文本
    final = next(e for e in events if e["type"] == "final_text")
    assert final["text"] == "修正后使用正确来源[1]。"
    assert gs.result.answer.grounded is True
    assert llm.complete_calls == 1, "应恰好调用一次修复"


def test_stream_degrades_to_extractive_when_repair_also_fails(upstream):
    """两轮都过不了校验 → 降级为原文摘录，并明确标记 grounded=False。

    宁可不流畅，也不给一个引用不可信的答案。
    """
    upstream([_delta("幻觉[9]。")])
    llm = FakeLLM(repair_text="还是幻觉[8]。")
    gs = AnswerGenerator(llm=llm).generate_stream("问题", [_hit(1, "第一段原文内容")])
    events = _drain(gs)

    assert any(e["type"] == "revoke" for e in events)
    verify = next(e for e in events if e["type"] == "verify")
    assert verify["grounded"] is False
    assert gs.result.answer.grounded is False
    assert "extractive" in gs.result.answer.degraded
    # 降级后引用必须 100% 来自原文
    assert gs.result.answer.citations[0].marker == 1
    assert "第一段原文内容" in gs.result.answer.text


def test_stream_without_llm_falls_back_without_any_delta():
    """没有模型时不该发任何 delta（因为根本没有逐字生成这回事），
    直接给抽取式答案。让前端误以为"在流式生成"是错的。"""

    class _NoLLM:
        enabled = False

    gs = AnswerGenerator(llm=_NoLLM()).generate_stream("问题", [_hit(1, "原文")])
    events = _drain(gs)
    assert not any(e["type"] == "delta" for e in events)
    assert gs.result.answer.degraded == "llm_unavailable_extractive"


def test_stream_with_no_hits_skips_model_entirely():
    class _Boom:
        enabled = True

        def stream(self, *a, **k):  # pragma: no cover
            raise AssertionError("召回为空时不应发起生成")

        def complete(self, *a, **k):  # pragma: no cover
            raise AssertionError("召回为空时不应发起生成")

    gs = AnswerGenerator(llm=_Boom()).generate_stream("问题", [])
    events = _drain(gs)
    assert gs.result.answer.route == "refuse"
    assert events[0]["type"] == "gen_mode"
    assert events[0]["mode"] == "no_hits"
