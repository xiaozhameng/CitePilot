"""意图分流：三档漏斗（规则 → 分类 → 兜底）。

对应一个经典设计问题：「规则、分类模型、LLM 路由三者怎么配合」。
这里的实现是把它落成真代码：

  L0 规则层    正则/关键词，<1ms，零成本 —— 覆盖寒暄、空输入、明显越界
  L1 分类层    LLM 输出 JSON 的 {route, confidence} —— 生产里这一层应该是
               微调过的 BERT 小模型（10~30ms），这里用 LLM 代替以便先跑通
  L2 兜底层    置信度低于阈值，**不硬猜**，走澄清式反问

为什么置信度分三档而不是两档？
  高（≥0.75）→ 按分类结果走
  中（0.45~0.75）→ **偏保守地走检索**：检索有出处、答案可被核对，
                    猜错只是答得不够好；而"直接答"没有依据，猜错就是幻觉
  低（<0.45）→ 反问澄清

这个"中档走检索"的偏移是个刻意的工程选择：**误判的代价不对称**。
让模型在不确定时多查一次资料，比让它自信地胡说便宜得多。

还有第四条规则，是实测之后补的：
  出口是 refuse / clarify（"不查资料"的那两个）时，**门槛要单独抬到 0.90**。
理由见下面 NEED_CERTAIN 的注释 —— 一句话是"越不做事，越要证据"。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .llm import LLM, LLMError
from .schema import TraceStep

# 允许的四条出口
ROUTES = ("direct", "rag", "clarify", "refuse")

HIGH = 0.75
LOW = 0.45

# "不查资料"的两个出口。它们和 rag 的关键区别是：**用户什么都拿不到。**
#   拒答 → 用户被挡在门外
#   反问 → 用户得多说一轮才知道系统到底能不能答
# 而误判成 rag 的代价只是"白检索一次"——检索有出处可核对，猜错最多是答得不够好。
# 代价如此不对称，门槛就不该一样。
_DENY_ROUTES = ("refuse", "clarify")

# 走上面那两个出口所需的置信度。
# 为什么是 0.9 而不是沿用 0.75？因为模型的**自报置信度校准很差**，实测：
# 同一条正经问题（"XX 项目有几个节点？"）连问三次，拿到过
# rag(0.60) / refuse(0.85) / rag(0.45) —— **0.85 的误判比 0.60 的正判还自信**。
# 也就是说这个数字本身不可信，只能用来做"相对比较"。既然不可信，
# 就把它花在代价最低的那条路上：不确定时去查资料，而不是把用户拒掉。
NEED_CERTAIN = 0.90

# L0 规则：这些不需要问模型
_GREET = re.compile(r"^(你好|您好|hi|hello|嗨|在吗|哈喽|早上好|晚上好|谢谢|多谢|感谢|再见|拜拜)[!！。.~～\s]*$", re.I)
_META = re.compile(r"^(你是谁|你叫什么|你能做什么|怎么用|帮助|help|介绍一下你自己)")

_SYSTEM = """你是一个意图分类器。用户在一个「支付领域知识库问答」系统里提问，你要把问题分到四类之一。

分类定义：
- "rag"：问支付领域的知识、概念、流程、监管、机构、清结算等，需要查资料回答。绝大多数问题都属于这一类。
- "direct"：寒暄、闲聊、问你身份能力、与支付知识无关的元问题，不需要查资料。
- "clarify"：问题过于模糊或缺少关键信息，无法判断用户到底想问什么（例如"那个怎么样"）。
- "refuse"：要求你做超范围的事，例如预测具体股票、给出法律意见、执行转账、编造不存在的数据。

只输出 JSON，不要任何解释、不要 markdown 围栏：
{"route": "rag", "confidence": 0.92, "reason": "问的是备付金的定义"}

confidence 是你对这个判断的把握，0~1 之间的小数。拿不准就给低分，不要勉强给高分。"""


@dataclass
class IntentResult:
    route: str
    confidence: float
    reason: str
    layer: str  # L0_rule | L1_llm | L2_fallback
    steps: list[TraceStep]


def _rule_based(question: str) -> IntentResult | None:
    q = (question or "").strip()
    if not q or len(q) < 2:
        return IntentResult("clarify", 1.0, "输入过短，无法判断意图", "L0_rule", [])
    if _GREET.match(q) or _META.match(q):
        return IntentResult("direct", 0.95, "寒暄或元问题，规则命中", "L0_rule", [])
    # 明显越界：让系统替用户做资金/法律决策
    if re.search(r"(帮我|替我)(转账|付款|下单|投资|买|卖)", q):
        return IntentResult("refuse", 0.9, "涉及代用户执行资金操作", "L0_rule", [])
    return None


def _heuristic_fallback(question: str) -> IntentResult:
    """LLM 不可用时的兜底：有疑问词就当知识问题去检索。

    这里的取向和阈值策略一致 —— 宁可多查一次，也不要凭空作答。
    """
    if re.search(r"(什么|怎么|如何|为什么|哪些|哪个|区别|是不是|能否|流程|步骤|多少|谁)", question):
        return IntentResult("rag", 0.5, "LLM 不可用，按疑问词启发式判定为知识问题", "L2_fallback", [])
    return IntentResult("clarify", 0.3, "LLM 不可用且无明显疑问词，保守反问", "L2_fallback", [])


class IntentRouter:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm or LLM()

    def classify(self, question: str) -> IntentResult:
        # L0：规则先过，命中就直接返回，一次模型调用都不花
        hit = _rule_based(question)
        if hit:
            return hit

        # L1：LLM 分类（生产里换成微调小模型，接口不变）
        t0 = time.perf_counter()
        try:
            data, res = self.llm.complete_json(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": question}],
                temperature=0.0,
                max_tokens=800,
            )
            route = str(data.get("route", "")).strip()
            conf = float(data.get("confidence", 0.0))
            if route not in ROUTES:
                route, conf = "rag", 0.4  # 模型给了非法枚举值 → 按低置信处理
            reason = str(data.get("reason", ""))[:80]
            ms = res.ms
            steps = [TraceStep(name="intent_llm", ms=ms,
                               prompt_tokens=res.prompt_tokens,
                               completion_tokens=res.completion_tokens,
                               meta={"route": route, "confidence": round(conf, 3)})]
            # L2：置信度分流
            #
            # 一句话概括这段的取向：**越"不做事"的决定，要求的证据越高。**
            # 所以先判 deny 类出口，再判置信度档位 —— 顺序不能反，
            # 反了就会先按普通档位把 refuse 放行，来不及拦。
            if route in _DENY_ROUTES:
                if conf >= NEED_CERTAIN:
                    return IntentResult(route, conf, reason, "L1_llm", steps)
                # 记进 trace：这条降级必须可观测，否则"为什么这个问题去检索了"
                # 在事后根本查不出来。
                steps[0].meta["downgraded_from"] = route
                return IntentResult(
                    "rag", conf,
                    f"{route} 需 {NEED_CERTAIN:.2f} 以上置信（当前 {conf:.2f}），保守走检索：{reason}",
                    "L1_llm", steps)

            if conf < LOW:
                return IntentResult("clarify", conf, f"置信度过低（{conf:.2f}）：{reason}", "L1_llm", steps)
            if conf < HIGH:
                # 中档：不确定时偏向检索，因为检索的答案有出处可核对
                return IntentResult("rag", conf, f"置信度中等（{conf:.2f}），保守走检索：{reason}", "L1_llm", steps)
            return IntentResult(route, conf, reason, "L1_llm", steps)
        except Exception as exc:  # noqa: BLE001
            ms = int((time.perf_counter() - t0) * 1000)
            fb = _heuristic_fallback(question)
            fb.steps = [TraceStep(name="intent_llm", ms=ms, meta={"error": str(exc)[:120]})]
            return fb
