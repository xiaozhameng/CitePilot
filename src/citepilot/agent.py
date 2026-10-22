"""Agent 主循环 —— 整个项目的枢纽。

为什么它够得上叫「Agent」而不是「RAG 接口」？
因为它在"检索 → 生成"之间插了一个**判断与回头**的环节：

  检索 → 判充分性 → 不够就改写查询重检（最多 N 轮）→ 够才生成

普通 RAG 是一条直线，检索到什么就生成什么；Agent 会自己判断
"我这回捞到的东西够不够"，不够就换个说法再捞一次。
这个"判断 + 回头"就是 ReAct 里 Thought → Action → Observation 的最小形态，
也是「Agent Loop 怎么防死循环、防无限工具调用」的关键所在。

防死循环靠三道刹车（都在下面的代码里）：
  1) 硬性轮次上限 max_rewrite（默认 2），数满即停
  2) 改写后的查询如果和上一轮相同 → 立刻停（再检也是同样的结果）
  3) 无召回结果 → 不进循环，直接走兜底分支

一轮检索的成本是「一次 embedding + 一次 BM25 + 一次裁判调用」，
所以轮次上限不是拍脑袋：每多一轮，P95 延迟大约 +1.2s、成本大约 +40%。
把它设成 2 是拿"多捞回一些"换"别把延迟做爆"的平衡点。
"""

from __future__ import annotations

import time

from .answer import AnswerGenerator
from .config import SETTINGS
from .index import SearchIndex, default_chunks_path
from .ingest import load_chunks
from .intent import IntentRouter
from .judge import Judge
from .llm import LLM
from .retriever import Retriever
from .schema import Answer, TraceStep

_CLARIFY_TMPL = (
    "我不太确定你想问的是哪一块。可以帮我补充一点信息吗？\n"
    "比如：你关心的是**某一类机构**（支付机构 / 清算机构 / 央行）、"
    "**某一段流程**（收单 / 清算 / 结算 / 对账），还是**某个具体概念**（备付金 / 断直连 / 二清）？"
)

_REFUSE_TMPL = (
    "这件事我不能替你做。我可以解释支付领域的概念和流程、帮你理清监管要求，"
    "但涉及真实资金操作、投资决策或法律意见，需要由持牌机构或专业人员处理。"
)


# ---------- 检索循环：非流式与流式共用同一份控制逻辑 ----------
class RetrieveLoop:
    """自纠正检索循环。

    为什么把它做成一个「可迭代对象」而不是普通方法？
    因为同一条循环要被两种消费方式用到：

      · CLI / 评测：只关心结果，同步跑完就行
      · HTTP 流式：每一轮检索、每一次裁判判定，都要实时推给前端

    如果写两份（一份同步、一份生成器），两份的刹车条件迟早会改歪一个
    —— 而刹车条件是防死循环的安全阀，改歪了就是线上事故。
    所以这里只有一份循环，只是"要不要把中间过程吐出来"不同：
    迭代它就拿到事件，迭代结束后从属性上读结果。
    """

    def __init__(self, pilot: "CitePilot", question: str, confidence: float,
                 strategy: str, use_rerank: bool) -> None:
        self._p = pilot
        self._q = question
        self._conf = confidence
        self.strategy = strategy
        self.use_rerank = use_rerank
        # 迭代结束后可读的结果
        self.hits: list = []
        self.rewrites = 0
        self.degraded = ""
        self.query = question
        self.steps: list[TraceStep] = []

    def __iter__(self):
        p, q = self._p, self._q
        query = q
        prev_query = ""
        steps = self.steps

        while True:
            rr = p.retriever.retrieve(query, strategy=self.strategy, use_rerank=self.use_rerank)
            steps.extend(rr.steps)
            if rr.degraded:
                self.degraded = (self.degraded + "+" + rr.degraded).strip("+")
            self.hits = rr.hits
            yield {"type": "stage", "name": "retrieve", "ms": sum(s.ms for s in rr.steps),
                   "meta": {"round": self.rewrites, "hits": len(rr.hits),
                            "strategy": self.strategy}}

            # 刹车 3：压根没召回，循环再多次也没用
            if not rr.hits:
                steps.append(TraceStep(name="sufficiency", ms=0,
                                       meta={"sufficient": False, "reason": "无召回结果",
                                             "round": self.rewrites}))
                yield {"type": "stage", "name": "sufficiency", "ms": 0,
                       "meta": {"sufficient": False, "reason": "无召回结果"}}
                break

            # 判断充分性。裁判本身也是一次模型调用，它的 token 必须算进成本 ——
            # 否则成本会被系统性低估（这一层占单次成本的比例不小，实测约 1/4）。
            t0 = time.perf_counter()
            j_in0, j_out0 = p.judge.prompt_tokens, p.judge.completion_tokens
            ok, reason, missing = p.judge.sufficiency(q, rr.hits)
            ms = int((time.perf_counter() - t0) * 1000)
            steps.append(TraceStep(name="sufficiency", ms=ms,
                                   prompt_tokens=p.judge.prompt_tokens - j_in0,
                                   completion_tokens=p.judge.completion_tokens - j_out0,
                                   meta={"sufficient": ok, "reason": reason,
                                         "round": self.rewrites}))
            yield {"type": "stage", "name": "sufficiency", "ms": ms,
                   "meta": {"sufficient": ok, "reason": reason, "round": self.rewrites,
                            "missing": missing[:80] if missing else ""}}

            if ok:
                break
            # 刹车 1：轮次用满就停，带着现有资料去回答
            if self.rewrites >= p.max_rewrite:
                self.degraded = (self.degraded + "+max_rewrite").strip("+")
                break

            # 改写查询再检
            t0 = time.perf_counter()
            j_in0, j_out0 = p.judge.prompt_tokens, p.judge.completion_tokens
            new_query = p.judge.rewrite(q, missing)
            ms = int((time.perf_counter() - t0) * 1000)
            steps.append(TraceStep(name="rewrite", ms=ms,
                                   prompt_tokens=p.judge.prompt_tokens - j_in0,
                                   completion_tokens=p.judge.completion_tokens - j_out0,
                                   meta={"from": query[:60], "to": new_query[:60]}))
            yield {"type": "stage", "name": "rewrite", "ms": ms,
                   "meta": {"to": new_query[:80]}}

            # 刹车 2：改写没有变化，再检也还是同一批结果
            if new_query.strip() == query.strip() or new_query.strip() == prev_query.strip():
                self.degraded = (self.degraded + "+rewrite_loop").strip("+")
                break
            prev_query, query = query, new_query
            self.rewrites += 1

        self.query = query


class AskStream:
    """流式主循环。与 CitePilot.ask() 走同一批组件，只是把中间过程吐出来。

    刻意不继承、不复制 ask() 的逻辑分支 —— 两条路径共用
    IntentRouter / RetrieveLoop / AnswerGenerator，
    这里只负责"把结果翻译成事件"。真正的判断逻辑一处都不重复。
    """

    def __init__(self, pilot: CitePilot, question: str, strategy: str, use_rerank: bool) -> None:
        self._p = pilot
        self._q = question
        self._strategy = strategy
        self._use_rerank = use_rerank
        self.answer: Answer | None = None
        self.revoked = False

    def __iter__(self):
        p, question = self._p, self._q
        t_all = time.perf_counter()
        steps: list[TraceStep] = []

        yield {"type": "start", "question": question}

        # ① 意图分流
        it = p.router.classify(question)
        steps.extend(it.steps)
        for s in it.steps:
            yield {"type": "stage", "name": s.name, "ms": s.ms, "meta": s.meta}
        yield {"type": "route", "route": it.route, "confidence": it.confidence}

        if it.route == "direct":
            ans = p._direct(question, it, steps)
            ans.steps = steps
            steps.append(TraceStep(name="total", ms=int((time.perf_counter() - t_all) * 1000)))
            ans.total_ms = int((time.perf_counter() - t_all) * 1000)
            self.answer = ans
            yield {"type": "final_text", "text": ans.text}
            yield {"type": "answer", "answer": ans.to_dict()}
            return

        if it.route in ("clarify", "refuse"):
            tmpl = _CLARIFY_TMPL if it.route == "clarify" else _REFUSE_TMPL
            ans = Answer(question=question, text=tmpl, route=it.route,
                         intent=it.route, intent_confidence=it.confidence)
            ans.steps = steps
            steps.append(TraceStep(name="total", ms=int((time.perf_counter() - t_all) * 1000)))
            ans.total_ms = int((time.perf_counter() - t_all) * 1000)
            self.answer = ans
            yield {"type": "final_text", "text": ans.text}
            yield {"type": "answer", "answer": ans.to_dict()}
            return

        # ② 自纠正检索循环（事件实时转发）
        loop = RetrieveLoop(p, question, it.confidence, self._strategy, self._use_rerank)
        for ev in loop:
            yield ev
        steps.extend(loop.steps)

        # ③ 空结果兜底
        if not loop.hits:
            ans = Answer(
                question=question,
                text="知识库里没有检索到相关内容。可以换个说法，或者直接告诉我是哪一章的疑问"
                     "（例如「清结算」「牌照」「分账」「对账」）。",
                route="refuse", intent=it.route, intent_confidence=it.confidence,
                rewrites=loop.rewrites, degraded=loop.degraded or "no_hits",
            )
            ans.steps = steps
            steps.append(TraceStep(name="total", ms=int((time.perf_counter() - t_all) * 1000)))
            ans.total_ms = int((time.perf_counter() - t_all) * 1000)
            self.answer = ans
            yield {"type": "final_text", "text": ans.text}
            yield {"type": "answer", "answer": ans.to_dict()}
            return

        # ④ 带引用生成（流式 + 校验）
        gs = p.generator.generate_stream(question, loop.hits)
        for ev in gs:
            if ev["type"] == "done":
                continue
            if ev["type"] == "revoke":
                self.revoked = True
                yield ev
                continue
            if ev["type"] == "stage":
                steps.append(TraceStep(name=ev["name"], ms=ev["ms"]))
                yield ev
                continue
            yield ev

        gen = gs.result
        assert gen is not None, "生成流未产出结果"
        steps.extend(gen.steps)

        ans = gen.answer
        ans.intent = it.route
        ans.intent_confidence = it.confidence
        ans.rewrites = loop.rewrites
        if loop.degraded:
            ans.degraded = (ans.degraded + "+" + loop.degraded).strip("+")
        ans.steps = steps
        # 引用落地：前端拿到 citations 才能渲染角标对应的出处卡片
        yield {"type": "sources", "citations": [c.to_dict() for c in ans.citations]}
        ans.total_ms = int((time.perf_counter() - t_all) * 1000)
        steps.append(TraceStep(name="total", ms=ans.total_ms))
        self.answer = ans
        yield {"type": "answer", "answer": ans.to_dict()}


class CitePilot:
    def __init__(
        self,
        index: SearchIndex | None = None,
        llm: LLM | None = None,
        top_k: int | None = None,
        max_rewrite: int | None = None,
    ) -> None:
        self.llm = llm or LLM()
        if index is None:
            chunks = load_chunks(default_chunks_path())
            index = SearchIndex(chunks)
        self.index = index
        self.retriever = Retriever(index, top_k=top_k or SETTINGS.top_k, rrf_k=SETTINGS.rrf_k)
        self.router = IntentRouter(self.llm)
        self.judge = Judge(self.llm)
        self.generator = AnswerGenerator(self.llm)
        self.max_rewrite = SETTINGS.max_rewrite if max_rewrite is None else max_rewrite
        self.history: list[dict] = []

    # ---------- 主入口 ----------
    def ask(self, question: str, strategy: str = "hybrid", use_rerank: bool = True) -> Answer:
        t_all = time.perf_counter()
        steps: list[TraceStep] = []

        # ① 意图分流
        it = self.router.classify(question)
        steps.extend(it.steps)

        if it.route == "direct":
            return self._finish(self._direct(question, it, steps), steps, t_all)

        if it.route == "clarify":
            return self._finish(
                Answer(question=question, text=_CLARIFY_TMPL, route="clarify",
                       intent=it.route, intent_confidence=it.confidence),
                steps, t_all,
            )

        if it.route == "refuse":
            return self._finish(
                Answer(question=question, text=_REFUSE_TMPL, route="refuse",
                       intent=it.route, intent_confidence=it.confidence),
                steps, t_all,
            )

        # ② 自纠正检索循环
        hits, rewrites, degraded, used_query = self._retrieve_with_correction(
            question, it.confidence, strategy, use_rerank, steps
        )

        # ③ 空结果兜底：不要拿空上下文去让模型编
        if not hits:
            return self._finish(
                Answer(
                    question=question,
                    text="知识库里没有检索到相关内容。可以换个说法，或者直接告诉我是哪一章的疑问"
                         "（例如「清结算」「牌照」「分账」「对账」）。",
                    route="refuse", intent=it.route, intent_confidence=it.confidence,
                    rewrites=rewrites, degraded=degraded or "no_hits",
                ),
                steps, t_all,
            )

        # ④ 带引用生成
        gen = self.generator.generate(question, hits)
        steps.extend(gen.steps)

        ans = gen.answer
        ans.intent = it.route
        ans.intent_confidence = it.confidence
        ans.rewrites = rewrites
        if degraded:
            ans.degraded = (ans.degraded + "+" + degraded).strip("+")
        ans.steps = steps
        return self._finish(ans, steps, t_all)

    # ---------- 流式入口 ----------
    def ask_stream(self, question: str, strategy: str = "hybrid",
                   use_rerank: bool = True) -> "AskStream":
        """流式问答。用法：

            s = pilot.ask_stream(q)
            for ev in s:
                push_to_client(ev)      # 事件立即转给前端
            ans = s.answer              # 迭代结束后取完整结果（含引用、trace）

        事件类型（Java 网关按这套协议透传）：
            start     开问
            stage     某一步完成（intent / retrieve / sufficiency / rewrite / generate）
            route     意图分流结果（rag / direct / clarify / refuse）
            thinking  推理模型的思考片段（不是正文，UI 应另置展示）
            delta     正文增量
            revoke    刚才流出去的内容没通过引用校验，前端应清屏
            final_text 最终正文（revoke 后一定会有；正常路径用于对齐）
            verify    引用校验结论
            answer    完整 Answer 的 JSON（含 citations / metrics）
            error     异常
        """
        return AskStream(self, question, strategy, use_rerank)

    # ---------- 自纠正检索 ----------
    def _retrieve_with_correction(
        self, question: str, confidence: float, strategy: str, use_rerank: bool,
        steps: list[TraceStep],
    ) -> tuple[list, int, str, str]:
        """非流式入口：把循环跑完，只取结果。控制逻辑全在 RetrieveLoop 里。"""
        loop = RetrieveLoop(self, question, confidence, strategy, use_rerank)
        for _ in loop:  # 事件在这里被丢弃，CLI / 评测不需要中间过程
            pass
        steps.extend(loop.steps)
        return loop.hits, loop.rewrites, loop.degraded, loop.query

    # ---------- 非检索路径 ----------
    def _direct(self, question: str, it, steps: list[TraceStep]) -> Answer:
        if not self.llm.enabled:
            return Answer(question=question,
                          text="（模型未配置，无法进行自由对话。我仍可以回答支付知识库里的问题。）",
                          route="direct", intent=it.route, intent_confidence=it.confidence)
        try:
            res = self.llm.complete(
                [{"role": "system", "content": "你是「支付通识」知识库的问答助手。用一两句中文简短回应。不要用你自己的知识回答支付领域的具体问题，那类问题请引导用户直接提问。"},
                 {"role": "user", "content": question}],
                temperature=0.3, max_tokens=200,
            )
            steps.append(TraceStep(name="direct_answer", ms=res.ms,
                                   prompt_tokens=res.prompt_tokens,
                                   completion_tokens=res.completion_tokens))
            return Answer(question=question, text=res.text, route="direct",
                          intent=it.route, intent_confidence=it.confidence)
        except Exception as exc:  # noqa: BLE001
            steps.append(TraceStep(name="direct_answer", ms=0, meta={"error": str(exc)[:120]}))
            return Answer(question=question, text="我们直接开始吧，你想问支付里的哪一块？",
                          route="direct", intent=it.route, intent_confidence=it.confidence)

    # ---------- 收尾 ----------
    @staticmethod
    def _finish(ans: Answer, steps: list[TraceStep], t_all: float) -> Answer:
        ans.steps = steps
        steps.append(TraceStep(name="total", ms=int((time.perf_counter() - t_all) * 1000)))
        return ans
