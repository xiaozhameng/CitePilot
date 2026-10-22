"""配置：只从环境变量与 .env 读，代码里零硬编码密钥。

为什么要单独一层？
因为「换模型 / 换 embedding 提供方」这件事必须做到只改配置不改代码 ——
这既是工程卫生，也是「模型供应商切换」的最小实现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件在 src/citepilot/config.py，往上三层）
ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


@dataclass(frozen=True)
class Settings:
    # 生成模型
    api_key: str
    api_base: str
    model: str

    # embedding
    embed_provider: str      # bm25 | ollama | siliconflow
    embed_model: str
    embed_base: str
    embed_api_key: str

    # rerank
    rerank_provider: str     # none | siliconflow
    rerank_model: str
    rerank_base: str
    rerank_api_key: str

    # 检索
    top_k: int
    rrf_k: int
    max_rewrite: int

    # 计价（元 / 百万 token）。默认值放在 .env 里，方便按当期价格改。
    price_in_per_m: float
    price_out_per_m: float

    # 路径
    root: Path
    corpus_dir: Path      # 内置语料（那本支付手册）
    kb_dir: Path          # 手动上传的知识库
    db_path: Path

    @property
    def has_llm(self) -> bool:
        """没有有效 key 时，Agent 自动降级到「只检索不生成」的抽取式回答。"""
        return bool(self.api_key) and not self.api_key.startswith("sk-xxx")

    def cost_cny(self, prompt_tokens: int, completion_tokens: int) -> float:
        """单次调用成本。成本数字必须能被拆解，否则解释不了"钱花在哪"。"""
        return (
            prompt_tokens / 1_000_000 * self.price_in_per_m
            + completion_tokens / 1_000_000 * self.price_out_per_m
        )


def load_settings() -> Settings:
    corpus = Path(_env("CITEPILOT_CORPUS_DIR", "corpus/payment-handbook"))
    kb = Path(_env("CITEPILOT_KB_DIR", "data/kb"))
    db = Path(_env("CITEPILOT_DB_PATH", "var/citepilot.db"))
    return Settings(
        api_key=_env("CITEPILOT_API_KEY"),
        api_base=_env("CITEPILOT_API_BASE", "https://api.deepseek.com/v1"),
        model=_env("CITEPILOT_MODEL", "openai/deepseek-v4-flash"),
        embed_provider=_env("CITEPILOT_EMBED_PROVIDER", "bm25"),
        embed_model=_env("CITEPILOT_EMBED_MODEL", "bge-m3"),
        embed_base=_env("CITEPILOT_EMBED_BASE", "http://127.0.0.1:11434/v1"),
        embed_api_key=_env("CITEPILOT_EMBED_API_KEY"),
        rerank_provider=_env("CITEPILOT_RERANK_PROVIDER", "none"),
        rerank_model=_env("CITEPILOT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_base=_env("CITEPILOT_RERANK_BASE", "https://api.siliconflow.cn/v1"),
        rerank_api_key=_env("CITEPILOT_RERANK_API_KEY"),
        top_k=int(_env("CITEPILOT_TOP_K", "8")),
        rrf_k=int(_env("CITEPILOT_RRF_K", "60")),
        max_rewrite=int(_env("CITEPILOT_MAX_REWRITE", "2")),
        price_in_per_m=float(_env("CITEPILOT_PRICE_IN_PER_M", "1.0")),
        price_out_per_m=float(_env("CITEPILOT_PRICE_OUT_PER_M", "2.0")),
        root=ROOT,
        corpus_dir=corpus if corpus.is_absolute() else ROOT / corpus,
        kb_dir=kb if kb.is_absolute() else ROOT / kb,
        db_path=db if db.is_absolute() else ROOT / db,
    )


SETTINGS = load_settings()
