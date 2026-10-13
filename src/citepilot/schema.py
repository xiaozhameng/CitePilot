"""统一数据结构。

为什么要单独一个文件？
因为「检索 → 融合 → 生成 → 引用校验 → 评测」这五步跨了好几个模块，
如果每个模块自己造一个 dict，字段名迟早会漂移（rank 一会儿叫 rank 一会儿叫 order），
最后错误会以「取不到字段」的形式在最远的地方爆出来。
这里把跨模块传递的四种对象钉死：Chunk（语料切片）、Hit（召回结果）、
Citation（引用）、TraceStep（单步耗时与 token）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
from urllib.parse import quote


@dataclass
class Chunk:
    """语料的最小检索单元。带足够的「出身信息」，引用才能精确到章节。"""

    chunk_id: str          # 全局唯一，格式 {chapter_id}::{section_idx}::{chunk_idx}
    chapter_id: str        # 章节锚点，如 ch-03，可直接拼成 HTML 深链
    chapter_no: str        # 章号显示名，如「第 3 章」「附录 A」
    chapter_title: str     # 章标题
    part: str              # 篇名，如「第一篇 · 基础与监管」
    section_title: str     # 节标题（H2），引用时显示到这一级
    text: str              # 正文
    kind: str = "prose"    # prose | figure_caption | term | qa
    order: int = 0         # 在全书中的顺序，用于稳定排序

    # ---- 文档维度：支持多份文档之后才有这三个字段 ----
    # 为什么必须加：只有「章节」概念时，两份文档的第 3 章会撞在一起，
    # 引用标签分不清「第 3 章」是哪本书的，anchor 也会指向错误的深链。
    doc_id: str = "payment-handbook"   # 来自哪份文档；内置语料固定用它
    doc_name: str = ""                 # 文档显示名；内置语料留空（标签里不加前缀）
    origin: str = "builtin"            # builtin | upload —— 决定 anchor 怎么生成

    @property
    def label(self) -> str:
        """人类可读的出处标签，直接进答案里的引用角标。

        两条分支不是冗余，是刻意的：
        · 上传的文档没有「章」的概念，出处就是「文档名 · 小节」；
        · 内置语料必须保持原格式不动 —— 57 条评测集、已有的引用展示、
          文档里的示例输出全都依赖「第 3 章 支付体系 · 3.1 ...」这个形状。
        """
        if self.origin == "upload":
            if self.section_title:
                return f"{self.doc_name} · {self.section_title}"
            return self.doc_name or "未命名文档"
        if self.section_title:
            return f"{self.chapter_no} {self.chapter_title} · {self.section_title}"
        return f"{self.chapter_no} {self.chapter_title}"

    @property
    def anchor(self) -> str:
        """引用深链：点一下就能跳到原文。

        内置语料有静态 HTML，直接给章节锚点；
        上传的文档没有可点的网页，所以指向内核的原文接口 —— 点开能看到那一段原文。
        （这个接口存在的唯一理由就是让「引用可追溯」这件事对上传文档同样成立。）
        """
        if self.origin == "upload":
            return f"/v1/kb/documents/{quote(self.doc_id)}/raw?chunk={quote(self.chunk_id)}"
        return f"corpus/payment-handbook/index.html#{self.chapter_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        d["anchor"] = self.anchor
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class Hit:
    """一条召回结果。分数拆开存，是为了让「对照实验」可解释。"""

    chunk: Chunk
    score: float = 0.0          # 融合后的最终分
    vec_rank: int | None = None  # 向量一路的排名（None 表示该路没召回）
    bm25_rank: int | None = None  # BM25 一路的排名
    rerank_score: float | None = None
    source: str = ""            # 最终名次由哪一路贡献（vec / bm25 / both）

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Citation:
    """答案里的一条引用。marker 是正文里的 [1][2] 角标。"""

    marker: int          # 角标编号，从 1 开始
    chunk_id: str
    label: str
    anchor: str
    quote: str = ""      # 引用到的原文片段（用于人工核对 + 忠实度判定）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceStep:
    """一步的耗时与 token。全链路可观测的最小粒度。"""

    name: str
    ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """一次问答的完整产出。CLI / 网关 / 评测三个消费方都读它。"""

    question: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    intent: str = ""
    intent_confidence: float = 0.0
    route: str = ""                  # direct | rag | clarify | refuse
    rewrites: int = 0                # 自纠正循环改写了几次
    grounded: bool = True            # 引用校验是否通过
    degraded: str = ""               # 走了哪条降级路径（空 = 未降级）
    steps: list[TraceStep] = field(default_factory=list)
    total_ms: int = 0                # 墙钟总耗时。单独存，不混进 steps，理由见 _timed_steps

    @property
    def _timed_steps(self) -> list[TraceStep]:
        """统计用的步骤序列：排除 total。

        total 是「整条链路的墙钟时间」，它本身就包含了其他所有步骤的耗时。
        如果把它一起加进 latency，同一个耗时会被算两遍 ——
        第一版就是这样：真实 5.6s 的问答报成了 12.7s。
        凡是"汇总项"，都不能再参与汇总。
        """
        return [s for s in self.steps if s.name != "total"]

    @property
    def latency_ms(self) -> int:
        return sum(s.ms for s in self._timed_steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self._timed_steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self._timed_steps)

    def to_dict(self, hits_limit: int = 8) -> dict[str, Any]:
        """对外序列化。HTTP 层直接吐这个。

        为什么 hits 要截断？
        一次召回可能有 24 条候选（top_k × RECALL_MULTIPLIER），
        全塞进响应体里，一个回答的 JSON 能到 100KB+。
        前端真正要用的是「最终展示的这几条」+「引用了哪几条」，
        所以这里只带前 hits_limit 条，其余靠 trace 库回查。
        """
        import dataclasses

        return {
            "question": self.question,
            "text": self.text,
            "route": self.route,
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "rewrites": self.rewrites,
            "grounded": self.grounded,
            "degraded": self.degraded,
            "citations": [c.to_dict() for c in self.citations],
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "label": h.chunk.label,
                    "anchor": h.chunk.anchor,
                    "score": round(h.score, 6),
                    "vec_rank": h.vec_rank,
                    "bm25_rank": h.bm25_rank,
                    "rerank_score": (round(h.rerank_score, 6)
                                     if h.rerank_score is not None else None),
                    "source": h.source,
                    "preview": h.chunk.text[:160],
                }
                for h in self.hits[:hits_limit]
            ],
            "metrics": {
                "latency_ms": self.latency_ms,
                "total_ms": self.total_ms or self.latency_ms,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "steps": [dataclasses.asdict(s) for s in self.steps],
            },
        }
