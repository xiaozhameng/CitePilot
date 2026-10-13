"""检索器：两路召回 → 去重 → RRF 融合 →（可选）精排 → Top-K。

这里把「策略」做成了参数，而不是写死：
因为对照实验需要同一份代码跑出不同配置的数字。
如果为了跑实验去改代码，那实验就没有可比性了 ——
同一个函数、同一份语料、同一个评测集，只有配置不同，数字才可信。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .embed import RerankClient
from .fusion import dedupe_by_text, rrf_fuse
from .index import SearchIndex
from .schema import Hit, TraceStep

# 每一路先各召回这么多，再融合。召回要"求全"，所以放大到 Top-K 的 2~3 倍。
RECALL_MULTIPLIER = 3


@dataclass
class RetrieveResult:
    hits: list[Hit] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    strategy: str = "hybrid"
    degraded: str = ""


class Retriever:
    def __init__(
        self,
        index: SearchIndex,
        reranker: RerankClient | None = None,
        top_k: int = 8,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.index = index
        self.reranker = reranker or RerankClient()
        self.top_k = top_k
        self.rrf_k = rrf_k
        # 默认两路等权。BM25 调低一点是因为它容易"字面命中"但语义跑偏。
        self.weights = weights or {"vec": 1.0, "bm25": 0.9}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str = "hybrid",
        use_rerank: bool = True,
    ) -> RetrieveResult:
        top_k = top_k or self.top_k
        pool = top_k * RECALL_MULTIPLIER
        steps: list[TraceStep] = []
        degraded = ""

        ranked: dict[str, list[str]] = {}

        # ---- 第一路：BM25（稀疏，永远在）----
        t0 = time.perf_counter()
        bm25_raw = self.index.search_bm25(query, k=pool)
        steps.append(TraceStep(name="recall_bm25", ms=int((time.perf_counter() - t0) * 1000),
                               meta={"n": len(bm25_raw)}))
        if strategy in {"hybrid", "bm25_only"}:
            ranked["bm25"] = dedupe_by_text(
                [(cid, self.index.by_id[cid].text) for cid, _ in bm25_raw]
            )

        # ---- 第二路：向量（稠密，可能不可用）----
        if strategy in {"hybrid", "vec_only"}:
            t0 = time.perf_counter()
            if self.index.vec_ready:
                vec_raw = self.index.search_vec(query, k=pool)
                ranked["vec"] = dedupe_by_text(
                    [(cid, self.index.by_id[cid].text) for cid, _ in vec_raw]
                )
                steps.append(TraceStep(name="recall_vec", ms=int((time.perf_counter() - t0) * 1000),
                                       meta={"n": len(vec_raw)}))
            else:
                degraded = "vec_unavailable"
                steps.append(TraceStep(name="recall_vec", ms=0, meta={"skipped": "向量索引不可用"}))
                if strategy == "vec_only":
                    # 只走向量但向量挂了 → 立刻降级到 BM25，而不是返回空
                    degraded = "vec_only->bm25"
                    ranked["bm25"] = dedupe_by_text(
                        [(cid, self.index.by_id[cid].text) for cid, _ in bm25_raw]
                    )

        if not ranked:
            return RetrieveResult(hits=[], steps=steps, strategy=strategy, degraded=degraded or "no_route")

        # ---- 融合 ----
        t0 = time.perf_counter()
        fused = rrf_fuse(ranked, weights=self.weights, k=self.rrf_k,
                         top_k=pool if use_rerank and self.reranker.enabled else top_k)
        hits: list[Hit] = []
        for cid, score, ranks in fused:
            hits.append(
                Hit(
                    chunk=self.index.by_id[cid],
                    score=score,
                    vec_rank=ranks.get("vec"),
                    bm25_rank=ranks.get("bm25"),
                    source="both" if len(ranks) == 2 else next(iter(ranks), ""),
                )
            )
        steps.append(TraceStep(name="fuse_rrf", ms=int((time.perf_counter() - t0) * 1000),
                               meta={"k": self.rrf_k, "pool": len(hits)}))

        # ---- 精排（cross-encoder）----
        if use_rerank and self.reranker.enabled and hits:
            t0 = time.perf_counter()
            try:
                pairs = self.reranker.rerank(query, [h.chunk.text for h in hits])
                by_idx = {i: s for i, s in pairs}
                for i, h in enumerate(hits):
                    h.rerank_score = by_idx.get(i)
                hits.sort(key=lambda h: -(h.rerank_score if h.rerank_score is not None else -1e9))
                steps.append(TraceStep(name="rerank", ms=int((time.perf_counter() - t0) * 1000),
                                       meta={"n": len(hits)}))
            except Exception as exc:  # noqa: BLE001
                # 精排失败不影响主链路：RRF 的名次已经是可用的排序
                degraded = (degraded + "+rerank_failed").strip("+")
                steps.append(TraceStep(name="rerank", ms=int((time.perf_counter() - t0) * 1000),
                                       meta={"error": str(exc)[:120]}))

        return RetrieveResult(hits=hits[:top_k], steps=steps, strategy=strategy, degraded=degraded)
