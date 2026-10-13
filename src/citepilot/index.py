"""检索索引：BM25 + 向量两路，以及「什么时候不该上向量数据库」的判断。

数据量说明一切：本书切完约 200 个 Chunk。
在这个量级上，暴力算余弦相似度是一次 200×1024 的矩阵乘，耗时微秒级，
比一次网络往返还快 —— 所以默认实现是**内存里的 numpy 暴力检索**，
而不是 pgvector / Milvus。

这不是偷懒，是成本判断：引入向量数据库会带来
「建索引 / 连库 / 版本升级 / 备份」四份运维成本，
换来的是"在百万级向量上把召回从 50ms 降到 5ms"。
语料只有 200 条时，这四份成本换不到任何东西。
真实项目里的切换阈值大约在 10 万条（在那之前 ANN 索引的召回损失反而更难调）。

当语料涨到那个量级，把 `PgVectorIndex`（见 pgstore.py）的开关打开即可，
上层检索代码一行不改。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bm25 import BM25, register_terms
from .config import SETTINGS
from .embed import EmbeddingClient, cosine_matrix
from .schema import Chunk


class SearchIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingClient | None = None,
        build_vectors: bool = True,
    ) -> None:
        self.chunks = chunks
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        # 把语料自带的术语表注入分词器 —— 让 jieba 知道这个领域里
        # "什么是一个词"。内置词典只覆盖最通用的那批，语料里的术语
        # （如"分账回退""单边账"）靠这一步补齐。
        register_terms(self._glossary_terms())
        self.bm25 = BM25.from_chunks(chunks)
        self.embedder = embedder or EmbeddingClient()
        self.matrix: np.ndarray | None = None
        self.vec_error: str = ""

        if build_vectors and self.embedder.enabled:
            try:
                texts = [self._index_text(c) for c in chunks]
                self.matrix = self.embedder.embed(texts)
            except Exception as exc:  # noqa: BLE001
                # 关键设计：向量建不起来不让系统挂掉，而是降级到 BM25 一路，
                # 并把原因记下来给 trace / 健康检查看。
                self.matrix = None
                self.vec_error = str(exc)

    @staticmethod
    def _index_text(c: Chunk) -> str:
        """送进 embedding 的文本 = 出处标签 + 正文。
        加上章节路径能让"第 3 章讲了什么"这类问题命中正确章节。"""
        return f"{c.label}\n{c.text}"

    def _glossary_terms(self) -> list[str]:
        """从词典章的块里抠出术语名，作为分词补充词表。

        词典章的块形如「二清 「二次清算」的简称，指…」——
        术语名就在块首那一小段。这里只取块首很短的片段做候选，
        避免把整段话当词加进去。
        """
        out: list[str] = []
        for c in self.chunks:
            if c.chapter_id != "ch-appx-glossary":
                continue
            head = "".join(c.text.split())[:12]
            for sep in ("「", "（", "("):
                if sep in head:
                    head = head.split(sep)[0]
            for part in head.split("/"):
                t = part.strip(" ·、,，")
                if 2 <= len(t) <= 8:
                    out.append(t)
        return out

    # ---- 两路检索 ----
    def search_bm25(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=k)

    def search_vec(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        q = self.embedder.embed([query], use_cache=True)[0]
        sims = cosine_matrix(q, self.matrix)
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in order]

    # ---- 状态 ----
    @property
    def vec_ready(self) -> bool:
        return self.matrix is not None

    def describe(self) -> str:
        parts = [f"{len(self.chunks)} 个 Chunk", "BM25 ✓"]
        parts.append(f"向量 ✓（{self.matrix.shape[1]} 维）" if self.vec_ready else f"向量 ✗（{self.vec_error or '未启用'}）")
        return " | ".join(parts)


def default_chunks_path() -> Path:
    return SETTINGS.corpus_dir / "chunks.jsonl"
