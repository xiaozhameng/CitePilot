"""Embedding 客户端：一个接口，三个实现，配置一行切换。

为什么要做抽象而不是直接写死一家？
因为 DeepSeek 不提供 embedding 接口，向量必须另找地方算，
而"另找地方"这件事有三种现实答案：
  bm25        —— 不算向量，词频兜底（零依赖，永不失败）
  ollama      —— Docker 本地跑（离线、免费、要下模型）
  siliconflow —— 云 API（免费额度、不占本地资源）
三种答案各有代价，而且随时可能切换（模型下线、额度用尽、断网）。
把它们藏在同一个接口后面，上层的检索代码一行都不用改 ——
这就是「模型网关 / 供应商切换」的最小可运行版本。

embedding 结果按「内容哈希」缓存到磁盘：
  1) 重复索引同一份语料不用重算（省时间也省钱）
  2) 换了模型，哈希前缀变了，缓存自动失效（不会新旧向量混用）
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from .config import SETTINGS


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.provider = (provider or SETTINGS.embed_provider).lower()
        self.model = model or SETTINGS.embed_model
        self.base = (base or SETTINGS.embed_base).rstrip("/")
        self.api_key = api_key if api_key is not None else SETTINGS.embed_api_key
        self.cache_dir = cache_dir or (SETTINGS.root / "var" / "embed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dim: int | None = None

    # ---- 对外能力声明 ----
    @property
    def enabled(self) -> bool:
        return self.provider in {"ollama", "siliconflow"}

    @property
    def cache_file(self) -> Path:
        """缓存文件名带上 provider+model+base，换模型即换缓存。"""
        tag = f"{self.provider}|{self.model}|{self.base}"
        h = hashlib.sha1(tag.encode()).hexdigest()[:10]
        return self.cache_dir / f"emb_{h}.npz"

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ---- 缓存 ----
    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_file.exists():
            return {}
        try:
            data = np.load(self.cache_file, allow_pickle=False)
            keys = data["keys"].tolist()
            vecs = data["vecs"]
            return {k: vecs[i].tolist() for i, k in enumerate(keys)}
        except Exception:
            # 缓存坏了就当没有，不能因为缓存问题让整个检索挂掉
            return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        if not cache:
            # 全部删空时要把文件也删掉，否则下次读回来的还是旧内容
            self.cache_file.unlink(missing_ok=True)
            return
        keys = list(cache.keys())
        vecs = np.asarray([cache[k] for k in keys], dtype=np.float32)
        np.savez_compressed(self.cache_file, keys=np.asarray(keys), vecs=vecs)

    # ---- 实际调用 ----
    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """ollama 与 siliconflow 都是 OpenAI 兼容的 /embeddings 协议，共用一条实现。"""
        url = f"{self.base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out: list[list[float]] = []
        # 分批：一次塞 200 条容易被网关拒，也更容易超时
        batch = 16
        for i in range(0, len(texts), batch):
            payload = {"model": self.model, "input": texts[i : i + batch]}
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code >= 400:
                        raise EmbeddingError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    data = resp.json().get("data", [])
                    # 有的实现不保证返回顺序，按 index 字段排一次
                    data.sort(key=lambda d: d.get("index", 0))
                    out.extend([d["embedding"] for d in data])
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise EmbeddingError(
                    f"embedding 调用失败（provider={self.provider} model={self.model}）：{last_err}"
                )
        return out

    def forget(self, texts: list[str]) -> int:
        """把指定的文本从缓存里删掉，返回删掉的条数。

        为什么需要它？
        因为评测时如果不删，第二个配置就会命中第一个配置留下的缓存 ——
        测出来的延迟是"热缓存"的延迟，比真实值低一个数量级，
        报告里的对比就成了假的。第一版就是这么错的：
        纯向量那组 221ms，混合那组只有 3ms，因为查询向量已经缓存过了。

        顺带说明：这个缓存本身是真实存在的优化（高频问题的检索成本接近 0），
        但**评测延迟必须测冷路径**，两件事不能混。
        """
        cache = self._load_cache()
        n = 0
        for t in texts:
            if cache.pop(self._key(t), None) is not None:
                n += 1
        if n:
            self._save_cache(cache)
        return n

    def embed(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not self.enabled:
            raise EmbeddingError(f"provider={self.provider} 不提供向量，请用 BM25 一路")
        cache = self._load_cache() if use_cache else {}
        todo = [t for t in texts if self._key(t) not in cache]
        if todo:
            fresh = self._embed_remote(todo)
            for t, v in zip(todo, fresh):
                cache[self._key(t)] = v
            if use_cache:
                self._save_cache(cache)
        vectors = np.asarray([cache[self._key(t)] for t in texts], dtype=np.float32)
        self.dim = int(vectors.shape[1])
        return vectors

    def health(self) -> tuple[bool, str]:
        """探活：给降级链用。返回 (是否可用, 说明)。"""
        if not self.enabled:
            return False, f"provider={self.provider} 未启用向量"
        try:
            self.embed(["探活"], use_cache=False)
            return True, f"{self.provider}/{self.model} 可用"
        except Exception as exc:  # noqa: BLE001
            return False, f"{self.provider}/{self.model} 不可用：{exc}"


def cosine_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """余弦相似度。向量都做了归一化，所以等价于点积。"""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return m @ q


class RerankClient:
    """Rerank（cross-encoder 精排）客户端。

    为什么单独一个类？
    因为 rerank 和 embedding 是两种完全不同的模型形态：
    embedding 是 bi-encoder（query 和文档分开编码，可以预计算，快但粗）；
    rerank 是 cross-encoder（query 和文档拼在一起过模型，慢但准）。
    没有 GPU 时可以用云端的免费 rerank，这正是
    「没有 GPU 怎么做精排」这个问题的答案之一。
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = (provider or SETTINGS.rerank_provider).lower()
        self.model = model or SETTINGS.rerank_model
        self.base = (base or SETTINGS.rerank_base).rstrip("/")
        self.api_key = api_key if api_key is not None else SETTINGS.rerank_api_key

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "none"} and bool(self.api_key)

    def rerank(self, query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        """返回 [(原始下标, 相关性分)]，按分数降序。"""
        if not self.enabled:
            raise RuntimeError("rerank 未启用")
        url = f"{self.base}/rerank"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "query": query, "documents": docs}
        if top_n:
            payload["top_n"] = top_n
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"rerank HTTP {resp.status_code}: {resp.text[:200]}")
        results = resp.json().get("results", [])
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]
