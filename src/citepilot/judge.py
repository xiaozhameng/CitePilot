"""LLM 作为裁判（LLM-as-judge）：两个用途，一个实现。

用途一：**检索充分性判定**（在线，每次问答都跑）
  Agent 拿到召回结果后必须先问一句"这些资料够不够回答这个问题"，
  不够就改写 query 重检 —— 这就是自纠正循环的判据。
  没有这一步，Agent 只是"检索 + 生成"的固定流水线，不是 Agent。

用途二：**忠实度与相关性打分**（离线，评测时跑）
  Faithfulness = 答案里的断言有多大比例能被检索到的原文支持。
  这是幻觉率的代理指标，也是"怎么定位幻觉"这个问题的量化答案。

为什么用 LLM 当裁判而不是人工标注？
因为评测要能**每次改代码都跑一遍**。人工标 40 条要半天，跑一次就过时了。
LLM 裁判有偏差（对长答案更宽容），所以用它做**相对比较**很可靠
（同一套标准比 A、B 两个版本），做**绝对打分**要谨慎 —— 这个区别值得单独说清。
"""

from __future__ import annotations

import json
import re

from .llm import LLM, LLMError
from .schema import Hit

_SUFFICIENCY = """你在判断【资料】是否足以回答【问题】。

规则：
- 只要资料里包含了回答问题所需的关键事实，就算 sufficient=true（措辞不同没关系）。
- 如果资料只沾边、缺少回答所必需的关键信息，算 sufficient=false。
- 如果资料与问题无关，sufficient=false。

只输出 JSON：
{"sufficient": true, "reason": "一句话说明", "missing": "缺什么，没有就填空字符串"}"""

_REWRITE = """把用户的问题改写成**更适合检索**的查询。

要求：
- 补全指代（"它""那个"要变成具体名词）
- 拆出核心检索词，去掉"请问""能不能说说"这类口水话
- 如果问题太泛，往用户可能真正关心的具体概念上收
- 只输出改写后的那一句查询，不要任何解释、不要引号"""

_FAITH = """你在评估一段【回答】里的事实性断言，能不能被【资料】支持。

只输出 JSON：
{"supported": 数字, "total": 数字, "unsupported_claims": ["不被支持的断言原文"]}

supported = 能被资料明确支持的断言数；total = 回答里事实性断言总数（不含"资料中未提及"这类元陈述，也不含角标本身）。"""

_RELEVANCY = """你在评估【回答】是否切题地回答了【问题】。

只输出 JSON：
{"score": 0.0, "reason": "一句话"}

score 范围 0~1：1 = 直接、完整地回答了问题；0.5 = 相关但没答到点上；0 = 答非所问。
注意：回答里注明"资料中没有相关内容"且确实是知识库缺资料，属于切题回答，给 0.8 以上。"""


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def rule_rewrite(question: str) -> str:
    """不用模型的查询改写：抽关键词。

    这是改写环节的降级路径。它当然不如模型改写聪明 ——
    它补不上指代（不知道"那个意思"指的是什么），也做不了同义替换，
    但它**零延迟、零成本、永不失败**，而且对 BM25 这条路几乎是无损的
    （BM25 本来吃的就是关键词）。

    什么时候会走到这里：模型改写超时（4s）。宁可拿一个"笨但不慢"的改写
    再检一次，也不要为了一次改写把整条问答的延迟拖到 30s 以上。
    """
    from .bm25 import tokenize

    toks = [t for t in tokenize(question) if len(t) > 1]
    seen: list[str] = []
    for t in toks:
        if t not in seen:
            seen.append(t)
    return " ".join(seen) if seen else question


class Judge:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm or LLM()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _ask(self, system: str, user: str, timeout: float | None = None,
             retries: int | None = None) -> tuple[dict, int, int]:
        res = self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=1200,
            timeout=timeout,
            retries=retries,
        )
        self.prompt_tokens += res.prompt_tokens
        self.completion_tokens += res.completion_tokens
        return _extract_json(res.text), res.prompt_tokens, res.completion_tokens

    # ---- 在线：检索是否够用 ----
    def sufficiency(self, question: str, hits: list[Hit]) -> tuple[bool, str, str]:
        if not hits:
            return False, "没有任何召回结果", "需要检索到相关内容"
        if not self.llm.enabled:
            # 没有裁判可用时，用规则兜一下：有命中就认为够用
            return True, "裁判不可用，按有召回结果处理", ""
        ctx = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
        try:
            # 5 秒硬超时 + 不重试。
            # 为什么？裁判调用每次约 1.5~4.9s，而它一旦在推理模型上"想太久"，
            # 会直接把端到端 P95 从 25s 拉到 45s（实测）。宁可用规则兜底，
            # 也不要让一次判断拖垮一次问答。
            data, _, _ = self._ask(_SUFFICIENCY, f"【资料】\n{ctx}\n\n【问题】\n{question}",
                                   timeout=5.0, retries=1)
        except (LLMError, Exception):  # noqa: BLE001
            return True, "裁判超时/失败，降级为规则判断（有召回即视为够用）", ""
        ok = bool(data.get("sufficient", True))
        return ok, str(data.get("reason", ""))[:120], str(data.get("missing", ""))[:120]

    # ---- 在线：改写查询 ----
    def rewrite(self, question: str, missing: str = "") -> str:
        user = question if not missing else f"原问题：{question}\n缺少的信息：{missing}"
        try:
            res = self.llm.complete(
                [{"role": "system", "content": _REWRITE}, {"role": "user", "content": user}],
                temperature=0.2, max_tokens=600, timeout=4.0, retries=1,
            )
        except Exception:  # noqa: BLE001
            return rule_rewrite(question)
        new_q = res.text.strip().strip('"').strip("'").split("\n")[0]
        return new_q or rule_rewrite(question)

    # ---- 离线：忠实度 ----
    def faithfulness(self, answer_text: str, hits: list[Hit]) -> tuple[float, list[str]]:
        if not self.llm.enabled:
            return float("nan"), []
        ctx = "\n\n".join(h.chunk.text for h in hits)
        try:
            data, _, _ = self._ask(_FAITH, f"【资料】\n{ctx}\n\n【回答】\n{answer_text}")
        except LLMError:
            return float("nan"), []
        total = int(data.get("total", 0) or 0)
        supported = int(data.get("supported", 0) or 0)
        bad = list(data.get("unsupported_claims", []) or [])
        if total <= 0:
            return float("nan"), bad
        return min(supported / total, 1.0), bad

    # ---- 离线：切题度 ----
    def answer_relevancy(self, question: str, answer_text: str) -> float:
        if not self.llm.enabled:
            return float("nan")
        try:
            data, _, _ = self._ask(_RELEVANCY, f"【问题】\n{question}\n\n【回答】\n{answer_text}")
        except LLMError:
            return float("nan")
        try:
            return max(0.0, min(float(data.get("score", 0.0)), 1.0))
        except (TypeError, ValueError):
            return float("nan")
