"""命令行入口。用 argparse 而不是 Typer，是为了少一个依赖 ——
这个项目要能在任何时候 `python -m citepilot.cli` 直接跑起来。

四个子命令，正好对应开发与演示的四个动作：
  doctor   体检：模型、向量、精排、语料各自能不能用（对应"降级链"是否生效）
  build    建索引：切语料 + 建 BM25 + 建向量
  ask      问一个问题（演示主用）
  stats    看聚合指标（P95 / 成本 / 降级分布）
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import SETTINGS
from .index import SearchIndex, default_chunks_path
from .ingest import build_chunks, extract_glossary, extract_quiz, load_chunks, save_chunks
from .schema import Answer
from .trace import TraceStore

console = Console()


def _load_or_build_index(rebuild: bool = False) -> SearchIndex:
    path = default_chunks_path()
    if rebuild or not path.exists():
        src = SETTINGS.corpus_dir / "index.html"
        chunks = build_chunks(src)
        save_chunks(chunks, path)
        console.print(f"[dim]已切分语料：{len(chunks)} 个 Chunk → {path}[/dim]")
    else:
        chunks = load_chunks(path)
    return SearchIndex(chunks)


# ---------------- doctor ----------------
def cmd_doctor(_args: argparse.Namespace) -> int:
    from .embed import EmbeddingClient, RerankClient
    from .llm import LLM

    t = Table(title="CitePilot 体检", show_lines=False)
    t.add_column("组件", style="bold")
    t.add_column("状态")
    t.add_column("说明")

    llm = LLM()
    if SETTINGS.has_llm:
        try:
            res = llm.complete([{"role": "user", "content": "只回复两个字：可用"}], max_tokens=200)
            t.add_row("生成模型", "[green]✓[/green]", f"{SETTINGS.model} · {res.ms}ms · {res.text[:20]}")
        except Exception as exc:  # noqa: BLE001
            t.add_row("生成模型", "[red]✗[/red]", str(exc)[:70])
    else:
        t.add_row("生成模型", "[yellow]未配置[/yellow]", "将降级为原文摘录模式")

    emb = EmbeddingClient()
    ok, msg = emb.health()
    t.add_row("Embedding", "[green]✓[/green]" if ok else "[yellow]降级[/yellow]",
              msg[:80] if ok else f"{msg[:60]} → 自动走 BM25 单路")

    rk = RerankClient()
    t.add_row("Rerank", "[green]✓[/green]" if rk.enabled else "[yellow]未启用[/yellow]",
              f"{rk.provider}/{rk.model}" if rk.enabled else "RRF 名次直接当最终排序")

    src = SETTINGS.corpus_dir / "index.html"
    if src.exists():
        quiz, glos = extract_quiz(src), extract_glossary(src)
        t.add_row("语料", "[green]✓[/green]", f"{src.name} · 自测题 {len(quiz)} 条 · 术语 {len(glos)} 条")
    else:
        t.add_row("语料", "[red]✗[/red]", f"缺少 {src}")

    cp = default_chunks_path()
    t.add_row("Chunk 索引", "[green]✓[/green]" if cp.exists() else "[yellow]未建[/yellow]",
              f"{cp.name}（先跑 build）" if cp.exists() else "先跑 build")

    console.print(t)
    return 0


# ---------------- build ----------------
def cmd_build(args: argparse.Namespace) -> int:
    src = SETTINGS.corpus_dir / "index.html"
    chunks = build_chunks(src)
    save_chunks(chunks, default_chunks_path())

    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    console.print(f"[green]语料切分完成[/green]：{len(chunks)} 个 Chunk {kinds}")

    idx = SearchIndex(chunks, build_vectors=not args.no_vectors)
    console.print(f"[green]{idx.describe()}[/green]")

    import json
    out = SETTINGS.root / "evals" / "goldenset.jsonl"
    from .evalkit import build_goldenset
    gold = build_goldenset(src, chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
    console.print(f"[green]评测集已生成[/green]：{len(gold)} 条 → {out}")
    return 0


# ---------------- ask ----------------
def _render_answer(ans: Answer) -> None:
    console.print(Panel(ans.text, title="回答", border_style="cyan", padding=(1, 2)))

    if ans.citations:
        t = Table(title="出处", show_lines=False)
        t.add_column("#", style="bold cyan", width=3)
        t.add_column("章节", style="bold")
        t.add_column("原文摘录", overflow="fold")
        for c in ans.citations:
            t.add_row(str(c.marker), c.label, c.quote[:110] + ("…" if len(c.quote) > 110 else ""))
        console.print(t)

    meta = Table(box=None, show_header=False, padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("意图", f"{ans.intent}（置信度 {ans.intent_confidence:.2f}）")
    meta.add_row("自纠正", f"改写 {ans.rewrites} 次")
    meta.add_row("引用可信", "是" if ans.grounded else "[red]否（已降级摘录）[/red]")
    if ans.degraded:
        meta.add_row("降级", f"[yellow]{ans.degraded}[/yellow]")
    meta.add_row("延迟", f"{ans.latency_ms} ms")
    meta.add_row("Token", f"入 {ans.prompt_tokens} / 出 {ans.completion_tokens}")
    meta.add_row("成本", f"¥{SETTINGS.cost_cny(ans.prompt_tokens, ans.completion_tokens):.5f}")
    console.print(meta)

    steps = Table(title="链路耗时", show_lines=False)
    steps.add_column("步骤")
    steps.add_column("ms", justify="right")
    steps.add_column("token", justify="right")
    for s in ans.steps:
        steps.add_row(s.name, str(s.ms), str(s.prompt_tokens + s.completion_tokens))
    console.print(steps)


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent import CitePilot

    idx = _load_or_build_index()
    console.print(f"[dim]{idx.describe()}[/dim]")
    pilot = CitePilot(index=idx)
    ans = pilot.ask(args.question, strategy=args.strategy, use_rerank=not args.no_rerank)
    _render_answer(ans)
    TraceStore().save(ans)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .agent import CitePilot

    idx = _load_or_build_index()
    console.print(f"[dim]{idx.describe()}[/dim]")
    pilot = CitePilot(index=idx)
    store = TraceStore()
    console.print("[bold]输入问题，空行或 Ctrl-D 退出。[/bold]")
    while True:
        try:
            q = console.input("[bold cyan]你 > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ans = pilot.ask(q)
        _render_answer(ans)
        store.save(ans)
    console.print("[dim]再见。[/dim]")
    return 0


# ---------------- stats ----------------
def cmd_stats(_args: argparse.Namespace) -> int:
    s = TraceStore().summary()
    if not s["total"]:
        console.print("[yellow]还没有任何问答记录，先跑几次 ask。[/yellow]")
        return 0
    store = TraceStore()
    t = Table(title=f"聚合指标（{s['total']} 次问答）")
    t.add_column("指标")
    t.add_column("值", justify="right")
    t.add_row("平均延迟", f"{s['avg_ms']:.0f} ms")
    t.add_row("最大延迟", f"{s['max_ms']:.0f} ms")
    t.add_row("P50", f"{store.percentile(50)} ms")
    t.add_row("P95", f"{store.percentile(95)} ms")
    t.add_row("输入 token 合计", f"{s['prompt_tokens']:,}")
    t.add_row("输出 token 合计", f"{s['completion_tokens']:,}")
    cost = SETTINGS.cost_cny(s["prompt_tokens"], s["completion_tokens"])
    t.add_row("合计成本", f"¥{cost:.4f}")
    t.add_row("单次平均成本", f"¥{cost / s['total']:.5f}")
    t.add_row("引用校验未通过", str(s["ungrounded"]))
    t.add_row("触发降级", str(s["degraded_n"]))
    t.add_row("自纠正改写合计", str(s["rewrites"]))
    console.print(t)

    if s["steps"]:
        st = Table(title="各步骤耗时")
        st.add_column("步骤")
        st.add_column("次数", justify="right")
        st.add_column("平均 ms", justify="right")
        st.add_column("最大 ms", justify="right")
        st.add_column("token", justify="right")
        for row in s["steps"]:
            st.add_row(row["name"], str(row["n"]), f"{row['avg_ms']:.0f}",
                       str(row["max_ms"]), f"{row['tokens'] or 0}")
        console.print(st)

    if s["degraded_breakdown"]:
        dt = Table(title="降级分布（这是降级链是否真生效的证据）")
        dt.add_column("降级路径")
        dt.add_column("次数", justify="right")
        for row in s["degraded_breakdown"]:
            dt.add_row(row["degraded"], str(row["n"]))
        console.print(dt)
    return 0


# ---------------- serve ----------------
def cmd_serve(args: argparse.Namespace) -> int:
    """把内核起成常驻服务。CLI 和 HTTP 两种用法共用同一套逻辑，
    区别只是"过程要不要给别人看"。"""
    import os

    from .server import serve

    if args.token:
        os.environ["CITEPILOT_INTERNAL_TOKEN"] = args.token
    serve(host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citepilot", description="带出处的支付通识知识问答 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="检查各组件可用性").set_defaults(func=cmd_doctor)

    b = sub.add_parser("build", help="切语料、建索引、生成评测集")
    b.add_argument("--no-vectors", action="store_true", help="只建 BM25，不建向量")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="问一个问题")
    a.add_argument("question")
    a.add_argument("--strategy", default="hybrid",
                   choices=["hybrid", "bm25_only", "vec_only"], help="召回策略（对照实验用）")
    a.add_argument("--no-rerank", action="store_true", help="关闭精排")
    a.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="交互式问答").set_defaults(func=cmd_chat)
    sub.add_parser("stats", help="查看聚合指标").set_defaults(func=cmd_stats)

    s = sub.add_parser("serve", help="把内核起成 HTTP + SSE 服务（供 Java 网关或网页调用）")
    s.add_argument("--host", default=None, help="默认 127.0.0.1，只监听本机")
    s.add_argument("--port", type=int, default=None, help="默认 8077")
    s.add_argument("--token", default=None, help="内部共享 token，网关请求时带上")
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
