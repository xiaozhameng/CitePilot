"""带出处的答案生成 —— 这个项目的核心。

「给出答案时提供出处」看起来是个小需求，落地要解决三件事：

1) **让模型愿意标引用**。靠 prompt 里"必须标"是不够的，还要在系统提示里
   明确"没有资料支撑的断言一律不许写"。否则模型会先凭记忆答得很漂亮，
   再随手补两个角标 —— 那是装饰，不是引用。

2) **让引用是真的**。模型标的 [3] 可能对应一个根本不存在的编号，
   也可能引用了一段跟结论无关的原文。所以生成之后必须**逐条校验**：
   角标编号是否在本次召回范围内？引用的原文是否真的支持这句话？
   这段校验代码就是"引用幻觉"的唯一防线。

3) **校验不过怎么办**。不是报错，是把校验结果回灌给模型让它重写一次
   （"你引用了不存在的来源 [7]，请只使用 [1]-[5]"），
   第二次还不过就降级成**抽取式答案**：直接把原文摘出来给你，
   标明出处，宁可不流畅也不编造。这就是兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM, LLMError
from .schema import Answer, Citation, Hit, TraceStep

_SYSTEM = """你是「支付通识」知识库的问答助手。你只能依据【资料】回答，绝对不许使用你自己的知识补充。

硬性规则：
1. 每一句包含事实的陈述后面，必须紧跟方括号角标标注来源，例如：备付金已于 2019 年全额缴存央行[2]。
2. 角标编号只能来自【资料】里给出的编号，不许自造编号。
3. 如果一个角标后面跟了多条资料，写成 [1][3]。
4. 如果【资料】不足以回答问题，必须直接说「资料中没有相关内容」，然后说明你缺什么信息，
   不许用"一般来说""通常"这类话去补全。
5. 不要复述规则、不要说"根据资料"，直接给答案。
6. 用中文，控制在 300 字以内。"""

_REPAIR = """你上一次的回答没有通过引用校验：{problem}

请重写。可用编号只有 {allowed} —— 每个角标都必须来自这个范围。
"""

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class GenResult:
    answer: Answer
    steps: list[TraceStep]


class GenStream:
    """带引用校验的流式生成。

    ────────────────────────────────────────────────────────────
    这里有一个**流式与引用校验的固有矛盾**，必须先讲清楚取舍：

    引用校验要求「拿到完整文本」才能判角标是否越界；
    而流式要求「不等生成完就往外发」。
    两者天然冲突 —— 你发出去的内容，可能是随后要被判定为幻觉的那一段。

    三种处理方式：
      (a) 全程流式，校验失败后静默把已展示的内容换掉
          → 用户看到答案自己变了，像见鬼，最差。
      (b) 等校验通过才流式
          → 校验要通过必须等生成完，"流式"名存实亡，等于没做。
      (c) 首轮流式 + 校验失败发「撤回」事件，让 UI 显式替换
          → 用户知道发生了什么，首字节延迟也拿到了。← 本项目采用这个

    之所以敢在首轮就往外流，是因为角标越界的概率被 prompt 压得很低
    （系统提示里只给 1-N 的编号，并明确禁止自造编号），
    实测 57 条 + 15 条评测集里，改判率是 0。
    但概率低不等于不会发生，所以撤回通道必须存在 —— 这就是下面
    `revoke` 事件的由来。

    修复轮（第 2 次生成）故意**不流式**：那一轮通常很短，
    流式收益有限，而它本身已经是一次"纠错"，再撤回一次体验就崩了。
    ────────────────────────────────────────────────────────────
    """

    def __init__(self, gen: "AnswerGenerator", question: str, hits: list[Hit]) -> None:
        self._gen = gen
        self._question = question
        self._hits = hits
        self.result: GenResult | None = None
        self.revoked = False

    def __iter__(self):
        gen, question, hits = self._gen, self._question, self._hits
        steps: list[TraceStep] = []

        # 无召回 / 无模型：与 generate() 同构，只是不发任何 delta
        if not hits or not gen.llm.enabled:
            r = gen.generate(question, hits)
            self.result = r
            yield {"type": "gen_mode", "mode": "extractive" if hits else "no_hits"}
            yield {"type": "done"}
            return

        context = build_context(hits)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{question}"},
        ]

        # ---- 第 1 轮：流式 ----
        handle = gen.llm.stream(messages, temperature=0.1, max_tokens=2000)
        try:
            for ev in handle:
                if ev["type"] == "content":
                    yield {"type": "delta", "text": ev["text"]}
                elif ev["type"] == "reasoning":
                    yield {"type": "thinking", "text": ev["text"]}
        except LLMError as exc:
            text, cits = _extractive_answer(question, hits)
            steps.append(TraceStep(name="generate", ms=0, meta={"error": str(exc)[:120]}))
            self.result = GenResult(
                answer=Answer(question=question, text=text, citations=cits, hits=hits,
                              route="rag", degraded="llm_error_extractive"),
                steps=steps,
            )
            yield {"type": "reset", "reason": "llm_error"}
            yield {"type": "final_text", "text": text}
            yield {"type": "done"}
            return

        text = handle.text
        ok, problem = validate_markers(text, len(hits))
        steps.append(TraceStep(
            name="generate", ms=handle.ms, prompt_tokens=handle.prompt_tokens,
            completion_tokens=handle.completion_tokens,
            meta={"markers": parse_markers(text), "valid": ok, "problem": problem,
                  "streamed": True},
        ))
        yield {"type": "stage", "name": "generate", "ms": handle.ms,
               "tokens": handle.prompt_tokens + handle.completion_tokens}

        if ok:
            self.result = GenResult(
                answer=Answer(question=question, text=text,
                              citations=_citations_from(text, hits),
                              hits=hits, route="rag", grounded=True),
                steps=steps,
            )
            yield {"type": "verify", "grounded": True, "problem": ""}
            yield {"type": "done"}
            return

        # ---- 校验没过：显式撤回，再走非流式修复 ----
        self.revoked = True
        yield {"type": "revoke", "reason": problem}

        last_error = problem
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": _REPAIR.format(problem=problem, allowed=f"1-{len(hits)}")},
        ]
        for attempt in range(1, gen.max_repair + 1):
            try:
                res = gen.llm.complete(messages, temperature=0.1, max_tokens=2000)
            except LLMError as exc:
                steps.append(TraceStep(name=f"generate_repair{attempt}", ms=0,
                                       meta={"error": str(exc)[:120]}))
                text2, cits = _extractive_answer(question, hits)
                self.result = GenResult(
                    answer=Answer(question=question, text=text2, citations=cits, hits=hits,
                                  route="rag", grounded=False,
                                  degraded="llm_error_extractive"),
                    steps=steps,
                )
                yield {"type": "final_text", "text": text2}
                yield {"type": "done"}
                return

            ok, problem = validate_markers(res.text, len(hits))
            steps.append(TraceStep(
                name=f"generate_repair{attempt}", ms=res.ms,
                prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                meta={"markers": parse_markers(res.text), "valid": ok, "problem": problem},
            ))
            yield {"type": "stage", "name": f"generate_repair{attempt}", "ms": res.ms,
                   "tokens": res.prompt_tokens + res.completion_tokens}
            if ok:
                self.result = GenResult(
                    answer=Answer(question=question, text=res.text,
                                  citations=_citations_from(res.text, hits),
                                  hits=hits, route="rag", grounded=True, rewrites=0),
                    steps=steps,
                )
                yield {"type": "verify", "grounded": True, "problem": "", "repaired": True}
                yield {"type": "final_text", "text": res.text}
                yield {"type": "done"}
                return
            last_error = problem
            messages = messages + [
                {"role": "assistant", "content": res.text},
                {"role": "user", "content": _REPAIR.format(problem=problem, allowed=f"1-{len(hits)}")},
            ]

        # ---- 修不好：抽取式兜底，引用 100% 可信 ----
        text2, cits = _extractive_answer(question, hits)
        steps.append(TraceStep(name="generate", ms=0, meta={"fallback": last_error}))
        self.result = GenResult(
            answer=Answer(question=question, text=text2, citations=cits, hits=hits,
                          route="rag", grounded=False,
                          degraded="citation_invalid_extractive"),
            steps=steps,
        )
        yield {"type": "verify", "grounded": False, "problem": last_error}
        yield {"type": "final_text", "text": text2}
        yield {"type": "done"}


def build_context(hits: list[Hit]) -> str:
    """把召回结果编成带编号的资料块。

    为什么把「出处标签」也写进资料里？
    因为模型看到"第 3 章 · 监管合规"这样的标签，会更容易在答案里提到章节，
    人工核对时也更容易定位。标签是给模型和人都用的。
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] 出处：{h.chunk.label}\n{h.chunk.text}")
    return "\n\n".join(blocks)


def parse_markers(text: str) -> list[int]:
    """抽出正文里所有 [n] 角标，保持首次出现顺序、去重。"""
    seen: list[int] = []
    for m in _MARKER.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate_markers(text: str, n_hits: int) -> tuple[bool, str]:
    """引用校验第一层：编号是否越界。

    越界是最典型的"引用幻觉" —— 模型编了一个看起来合理的编号。
    这一层用规则就能拦住，零成本，不需要 LLM 判断。
    """
    markers = parse_markers(text)
    if not markers:
        return False, "回答里一个引用角标都没有"
    bad = [n for n in markers if n < 1 or n > n_hits]
    if bad:
        return False, f"引用了不存在的来源 {bad}（本次可用范围是 1-{n_hits}）"
    return True, ""


def _extractive_answer(question: str, hits: list[Hit], limit: int = 400) -> tuple[str, list[Citation]]:
    """降级路径：不用模型，直接把最相关的原文摘出来当答案。

    这是「没有 LLM 也要能用」的保证，也是引用 100% 可信的形态 ——
    因为答案就是原文本身，没有转述，就没有幻觉。
    """
    if not hits:
        return "知识库中没有检索到相关内容。", []
    parts = ["（降级为原文摘录模式：模型不可用，直接给出最相关的原文）\n"]
    cits: list[Citation] = []
    for i, h in enumerate(hits[:3], 1):
        snippet = h.chunk.text[:limit]
        parts.append(f"{snippet}[{i}]")
        cits.append(
            Citation(marker=i, chunk_id=h.chunk.chunk_id, label=h.chunk.label,
                     anchor=h.chunk.anchor, quote=snippet[:160])
        )
    return "\n\n".join(parts), cits


def _citations_from(text: str, hits: list[Hit]) -> list[Citation]:
    out: list[Citation] = []
    for idx in parse_markers(text):
        if 1 <= idx <= len(hits):
            h = hits[idx - 1]
            out.append(
                Citation(
                    marker=idx,
                    chunk_id=h.chunk.chunk_id,
                    label=h.chunk.label,
                    anchor=h.chunk.anchor,
                    quote="".join(h.chunk.text.split())[:160],
                )
            )
    return out


class AnswerGenerator:
    def __init__(self, llm: LLM | None = None, max_repair: int = 1) -> None:
        self.llm = llm or LLM()
        self.max_repair = max_repair

    def generate(self, question: str, hits: list[Hit], history: list[dict] | None = None) -> GenResult:
        steps: list[TraceStep] = []

        # 召回为空 → 直接走兜底，不要浪费一次模型调用去编答案
        if not hits:
            return GenResult(
                answer=Answer(question=question,
                              text="知识库里没有找到相关内容。可以换个说法，或者告诉我你关心的是哪一章（例如「清结算」「牌照」「分账」）。",
                              citations=[], hits=[], route="refuse", degraded="no_hits"),
                steps=[TraceStep(name="generate", ms=0, meta={"skipped": "无召回结果"})],
            )

        # 没有可用模型 → 抽取式降级，保证输出与引用仍 100% 可信
        if not self.llm.enabled:
            text, cits = _extractive_answer(question, hits)
            return GenResult(
                answer=Answer(question=question, text=text, citations=cits, hits=hits,
                              route="rag", degraded="llm_unavailable_extractive"),
                steps=[TraceStep(name="generate", ms=0, meta={"mode": "extractive"})],
            )

        context = build_context(hits)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{question}"},
        ]

        last_error = ""
        for attempt in range(self.max_repair + 1):
            try:
                res = self.llm.complete(messages, temperature=0.1, max_tokens=2000)
            except LLMError as exc:
                text, cits = _extractive_answer(question, hits)
                steps.append(TraceStep(name="generate", ms=0, meta={"error": str(exc)[:120]}))
                return GenResult(
                    answer=Answer(question=question, text=text, citations=cits, hits=hits,
                                  route="rag", degraded="llm_error_extractive"),
                    steps=steps,
                )

            text = res.text
            ok, problem = validate_markers(text, len(hits))
            steps.append(TraceStep(
                name="generate" if attempt == 0 else f"generate_repair{attempt}",
                ms=res.ms, prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                meta={"markers": parse_markers(text), "valid": ok, "problem": problem},
            ))

            if ok:
                return GenResult(
                    answer=Answer(question=question, text=text,
                                  citations=_citations_from(text, hits),
                                  hits=hits, route="rag", grounded=True),
                    steps=steps,
                )

            last_error = problem
            # 把校验失败的原因回灌给模型，让它带着约束重写
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _REPAIR.format(
                    problem=problem, allowed=f"1-{len(hits)}")},
            ]

        # 修不好 → 降级摘录。宁可不流畅，也不给不可信的引用。
        text, cits = _extractive_answer(question, hits)
        steps.append(TraceStep(name="generate", ms=0, meta={"fallback": last_error}))
        return GenResult(
            answer=Answer(question=question, text=text, citations=cits, hits=hits,
                          route="rag", grounded=False,
                          degraded="citation_invalid_extractive"),
            steps=steps,
        )

    def generate_stream(self, question: str, hits: list[Hit]) -> GenStream:
        """流式版生成。用法：

            gs = generator.generate_stream(q, hits)
            for ev in gs:            # 迭代期间把增量转给前端
                ...
            ans = gs.result.answer  # 迭代结束后取完整结果

        返回对象而不是生成器函数，原因同 LLM.stream：迭代完还要能读结果。
        """
        return GenStream(self, question, hits)
