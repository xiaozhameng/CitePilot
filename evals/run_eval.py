"""对照实验：同一份语料、同一个评测集，只改一个变量，出数字。

三组实验，每组只回答一个"为什么"：

  实验 A  纯 BM25  vs  向量+BM25+RRF      —— 混合召回到底值不值？
  实验 B  单轮检索  vs  自纠正重检（最多 2 轮）—— 多花的那次检索有没有回报？
  实验 C  无精排    vs  有 rerank          —— cross-encoder 的收益能不能覆盖它的延迟？

每一组都会同时报「质量指标（Recall/MRR/NDCG）」和「代价指标（延迟/token/成本）」。
只报质量不报代价，是最容易被追问的地方 —— 下一个问题一定是"那它的成本呢"。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citepilot.config import SETTINGS                    # noqa: E402
from citepilot.evalkit import (                          # noqa: E402
    load_goldenset, mrr, ndcg_at_k, recall_at_k,
)
from citepilot.index import SearchIndex, default_chunks_path  # noqa: E402
from citepilot.ingest import load_chunks                 # noqa: E402
from citepilot.retriever import Retriever                # noqa: E402

K_LIST = (1, 3, 5, 10)


def _chapter_hits(hits, k: int) -> list[str]:
    return [h.chunk.chapter_id for h in hits[:k]]


def eval_retrieval(retriever: Retriever, gold, strategy: str, use_rerank: bool,
                   top_k: int = 10) -> dict:
    hits_by_k: dict[int, list[list[str]]] = {k: [] for k in K_LIST}
    golds: list[list[str]] = []
    lat: list[int] = []

    for g in gold:
        t0 = time.perf_counter()
        rr = retriever.retrieve(g.question, top_k=top_k, strategy=strategy, use_rerank=use_rerank)
        lat.append(int((time.perf_counter() - t0) * 1000))
        golds.append(g.gold_chapters)
        for k in K_LIST:
            hits_by_k[k].append(_chapter_hits(rr.hits, k))

    out = {f"recall@{k}": recall_at_k(hits_by_k[k], golds, k) for k in K_LIST}
    out["mrr"] = mrr(hits_by_k[10], golds, 10)
    out["ndcg@10"] = ndcg_at_k(hits_by_k[10], golds, 10)
    out["latency_p50_ms"] = int(statistics.median(lat)) if lat else 0
    out["latency_p95_ms"] = int(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]) if lat else 0
    return out


def end_to_end(pilot, gold, max_rewrite: int, limit: int | None = None) -> dict:
    """端到端实验：跑完整 Agent（含裁判与生成），统计质量 + 代价。

    这一组会真的花钱（每次问答 2~4 次模型调用），所以支持 limit 限制样本数。
    """
    pilot.max_rewrite = max_rewrite
    subset = gold[:limit] if limit else gold
    rec, lat, tok_in, tok_out, rewrites, grounded = [], [], [], [], [], 0
    for g in subset:
        ans = pilot.ask(g.question)
        rec.append(set(_chapter_hits(ans.hits, 10)) & set(g.gold_chapters))
        lat.append(ans.latency_ms)
        tok_in.append(ans.prompt_tokens)
        tok_out.append(ans.completion_tokens)
        rewrites.append(ans.rewrites)
        grounded += 1 if ans.grounded else 0
    n = max(len(subset), 1)
    cin, cout = sum(tok_in), sum(tok_out)
    return {
        "n": len(subset),
        "chapter_recall@10": sum(1 for r in rec if r) / n,
        "avg_latency_ms": int(sum(lat) / n),
        "p95_latency_ms": int(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]) if lat else 0,
        "avg_rewrites": round(sum(rewrites) / n, 2),
        "cost_total_cny": round(SETTINGS.cost_cny(cin, cout), 4),
        "cost_per_query_cny": round(SETTINGS.cost_cny(cin, cout) / n, 5),
        "grounded_rate": round(grounded / n, 3),
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    if not rows:
        return
    cols = list(rows[0][1].keys())
    width = max(len(c) for c in cols) + 2
    print(f"\n=== {title} ===")
    header = "配置".ljust(30) + "".join(c.rjust(width) for c in cols)
    print(header)
    print("-" * len(header))
    for name, data in rows:
        print(name.ljust(30) + "".join(_fmt(data.get(c, "")).rjust(width) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="端到端实验限制多少条（0=全部）")
    ap.add_argument("--skip-e2e", action="store_true", help="只跑检索指标，不花模型费用")
    ap.add_argument("--gold", default="evals/goldenset.jsonl", help="评测集路径")
    ap.add_argument("--out", default="evals/report.json")
    args = ap.parse_args()

    chunks = load_chunks(default_chunks_path())
    index = SearchIndex(chunks)
    print(f"索引状态：{index.describe()}")

    gold = load_goldenset(SETTINGS.root / args.gold)
    for g in gold:
        g.gold_chapters = [c for c in g.gold_chapters]
    print(f"评测集：{args.gold} 共 {len(gold)} 条")
    report: dict = {"index": index.describe(), "n_gold": len(gold), "gold": args.gold}
    retriever = Retriever(index)

    # 评测延迟必须测「冷路径」：先把查询向量的缓存清掉，
    # 否则第二个配置会白蹭第一个配置的缓存，测出来的是热缓存延迟。
    queries = [g.question for g in gold]

    # ---------- 实验 A：召回策略 ----------
    rows = []
    for strategy, use_rerank, name in [
        ("bm25_only", False, "A1 纯 BM25"),
        ("vec_only", False, "A2 纯向量"),
        ("hybrid", False, "A3 向量+BM25+RRF"),
        ("hybrid", True, "A4 A3+精排(rerank)"),
    ]:
        if strategy == "vec_only" and not index.vec_ready:
            continue
        index.embedder.forget(queries)
        r = eval_retrieval(retriever, gold, strategy, use_rerank)
        rows.append((name, r))
        report.setdefault("expA_recall", {})[name] = r
    print_table("实验 A：不同召回策略（离线检索指标，不花模型费用）", rows)

    # ---------- 实验 C：精排（在最佳召回之上） ----------
    if index.vec_ready:
        rows = [
            ("C1 融合后不精排", report["expA_recall"].get("A3 向量+BM25+RRF", {})),
            ("C2 融合后精排", report["expA_recall"].get("A4 A3+精排(rerank)", {})),
        ]
        print_table("实验 C：精排的增益与代价", rows)

    # ---------- 实验 B：自纠正循环（端到端） ----------
    if not args.skip_e2e:
        from citepilot.agent import CitePilot
        pilot = CitePilot(index=index)
        rows = []
        limit = args.limit or None
        for mw, name in [(0, "B1 单轮检索"), (2, "B2 自纠正重检(≤2轮)")]:
            r = end_to_end(pilot, gold, max_rewrite=mw, limit=limit)
            rows.append((name, r))
            report.setdefault("expB_correction", {})[name] = r
        print_table("实验 B：自纠正检索循环（端到端，含生成成本）", rows)

    out = SETTINGS.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
