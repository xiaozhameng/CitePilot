"""HTTP 服务层 —— 内核对外的唯一入口。

────────────────────────────────────────────────────────────────
为什么一个 CLI 项目还要再包一层 HTTP？
────────────────────────────────────────────────────────────────
因为「内核」和「界面」必须分开，而且现在就要分开。

如果把问答逻辑直接写进 CLI，将来做桌面端、手机端、网页端的时候，
要么复制一遍逻辑（三份代码各自腐烂），要么把 CLI 当库 import
（界面进程一崩，正在跑的问答一起死）。

所以定下来的形态是：
    Python 内核 = 一个常驻进程，只认 HTTP + SSE
    所有界面 = 薄客户端，只会"发请求、画事件"

这条线看似多写了一个文件，但它换来三件事：
  1) 内核能脱离任何 UI 独立运行（`citepilot serve` 起来就是服务）
  2) 已经有 Java 网关来接它 —— 网关不 import 任何 Python
  3) 将来加一个手机端，不需要动内核一行代码

────────────────────────────────────────────────────────────────
三条工程约定
────────────────────────────────────────────────────────────────
1) **索引只加载一次**。SearchIndex 的构造要跑 jieba 分词建 BM25、
   还要载入 118 条向量，大约几百毫秒。每个请求重建等于自残。
   所以放在 lifespan 里做成单例；CitePilot 本身很轻（只是引用组件），
   每请求新建，避免多请求共享 history 之类的可变状态。

2) **流式端点用同步 def，不用 async def**。
   ask_stream() 是同步生成器，里面全是阻塞调用（检索、模型 API）。
   如果写成 async def，这些阻塞会把整个事件循环卡死，
   并发能力直接从"多路复用"掉到"一次一个"。
   写成 def，FastAPI 自动丢进线程池跑，这是最省事且正确的做法。

3) **内核不裸奔**。绑 127.0.0.1，且要求网关带上共享 token。
   多一层 token 看似多余（本机嘛），但它把"谁在调"这件事变得可审计 ——
   将来内核要挪到另一台机器上，改的只是一个地址。
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Iterator

from .config import SETTINGS

# 只在装了 web 依赖时才 import，避免没装的人连 CLI 都用不了
try:
    from fastapi import FastAPI, Header, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


# ------------------------------------------------------------------
# 单例：索引只建一次
# ------------------------------------------------------------------
class _Runtime:
    """进程级运行时。索引用懒加载 —— 第一次请求才建，`serve` 启动更快。"""

    def __init__(self) -> None:
        self._index = None
        self._lock = threading.Lock()
        self._trace_lock = threading.Lock()  # sqlite 多线程写要串行
        self.started_at = time.time()
        self.requests = 0
        self.errors = 0

    @property
    def index(self):
        if self._index is None:
            with self._lock:
                if self._index is None:  # 双检锁：避免两个请求同时建索引
                    from .index import SearchIndex, default_chunks_path
                    from .ingest import load_chunks

                    chunks = load_chunks(default_chunks_path())
                    self._index = SearchIndex(chunks)
        return self._index

    def new_pilot(self):
        """每个请求一个新的 CitePilot：它持有 history 等可变状态，不能共享。"""
        from .agent import CitePilot

        return CitePilot(index=self.index)

    def save_trace(self, ans) -> int:
        from .trace import TraceStore

        with self._trace_lock:
            return TraceStore().save(ans)


RT = _Runtime()


# ------------------------------------------------------------------
# SSE 编码
# ------------------------------------------------------------------
def sse(event: str, data: dict[str, Any]) -> str:
    """把一条事件编成 SSE 帧。

    格式上有两个细节必须对：
      · 事件名与数据之间是 `event: X\\ndata: Y\\n\\n`，**结尾必须是两个换行**
        （一个换行只是字段分隔，两个才是"这条消息结束"）。
      · data 里不能出现裸换行 —— 用 json.dumps 序列化就天然是安全的，
        字符串里的换行会被转义成 \\n。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _auth(token: str | None) -> None:
    """共享 token 校验。没配 token 就放行（纯本机调试场景）。"""
    expected = os.getenv("CITEPILOT_INTERNAL_TOKEN", "").strip()
    if not expected:
        return
    if (token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="内部 token 不正确")


def create_app() -> "FastAPI":
    if not _HAS_FASTAPI:  # pragma: no cover
        raise RuntimeError(
            "缺少 web 依赖。请先安装：pip install 'citepilot[server]' "
            "（即 fastapi + uvicorn）"
        )

    app = FastAPI(
        title="CitePilot Kernel",
        version="0.1.0",
        description="带出处的知识问答 Agent 内核。所有界面（CLI / Java 网关 / 网页）都是它的薄客户端。",
    )

    # 本机调试方便：允许任意来源。生产应改成白名单。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- 健康检查 ----------------
    @app.get("/health")
    def health(x_internal_token: str | None = Header(default=None)) -> dict:
        _auth(x_internal_token)
        idx = RT.index
        return {
            "ok": True,
            "uptime_s": round(time.time() - RT.started_at, 1),
            "requests": RT.requests,
            "errors": RT.errors,
            "index": idx.describe(),
            "llm": {
                "configured": SETTINGS.has_llm,
                "model": SETTINGS.model,
            },
            "embed": {
                "provider": SETTINGS.embed_provider,
                "model": SETTINGS.embed_model,
            },
            "rerank": {
                "provider": SETTINGS.rerank_provider,
                "model": SETTINGS.rerank_model,
            },
        }

    # ---------------- 非流式问答 ----------------
    @app.post("/v1/ask")
    def ask(body: dict, x_internal_token: str | None = Header(default=None)) -> dict:
        """一次问答，等全部算完再返回。

        什么时候用它？批量评测、脚本调用、以及任何"我不需要看到过程"的场景。
        交互式界面应该走 /v1/ask/stream —— 这条路径下用户要盯着空屏 5~10 秒。
        """
        _auth(x_internal_token)
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空")

        RT.requests += 1
        try:
            pilot = RT.new_pilot()
            ans = pilot.ask(
                question,
                strategy=body.get("strategy", "hybrid"),
                use_rerank=bool(body.get("use_rerank", True)),
            )
            ans.total_ms = ans.latency_ms
            trace_id = RT.save_trace(ans)
            out = ans.to_dict()
            out["trace_id"] = trace_id
            return out
        except Exception as exc:  # noqa: BLE001
            RT.errors += 1
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    # ---------------- 流式问答（SSE） ----------------
    @app.post("/v1/ask/stream")
    def ask_stream(body: dict, x_internal_token: str | None = Header(default=None)):
        """边算边推。

        注意这是个 **同步 def**（不是 async def），理由见文件头第 2 条：
        ask_stream 里全是阻塞调用，写成 async 会把事件循环卡死，
        并发能力从"多路复用"掉到"一次一个"。
        交给 FastAPI 的线程池，才是正确用法。
        """
        _auth(x_internal_token)
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question 不能为空")

        strategy = body.get("strategy", "hybrid")
        use_rerank = bool(body.get("use_rerank", True))
        session_id = body.get("session_id") or ""

        def gen() -> Iterator[str]:
            RT.requests += 1
            pilot = RT.new_pilot()
            t0 = time.perf_counter()
            # 首帧先发：让客户端立刻知道"连上了、开始干活了"，
            # 而不是对着一个没有任何输出的连接猜是不是挂了。
            yield sse("open", {"session_id": session_id, "ts": time.time()})
            try:
                stream = pilot.ask_stream(question, strategy=strategy, use_rerank=use_rerank)
                for ev in stream:
                    etype = ev.get("type", "message")
                    if etype == "answer":
                        # 完整答案落库，再把事件发出去
                        trace_id = RT.save_trace(stream.answer)
                        ev = dict(ev)
                        ev["trace_id"] = trace_id
                        ev["server_ms"] = int((time.perf_counter() - t0) * 1000)
                    yield sse(etype, {k: v for k, v in ev.items() if k != "type"})
            except Exception as exc:  # noqa: BLE001
                RT.errors += 1
                yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # 关掉 nginx 之类中间层的缓冲，否则流式会被攒成一坨再吐
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------- 聚合指标 ----------------
    @app.get("/v1/stats")
    def stats(x_internal_token: str | None = Header(default=None)) -> dict:
        _auth(x_internal_token)
        from .trace import TraceStore

        store = TraceStore()
        s = store.summary()
        s["p50_ms"] = store.percentile(50)
        s["p95_ms"] = store.percentile(95)
        s["cost_cny"] = SETTINGS.cost_cny(s["prompt_tokens"], s["completion_tokens"])
        if s["total"]:
            s["cost_per_ask_cny"] = round(s["cost_cny"] / s["total"], 6)
        return s

    # ---------------- 最近问答（演示页的"审计"用） ----------------
    @app.get("/v1/recent")
    def recent(limit: int = Query(default=20, ge=1, le=200),
               x_internal_token: str | None = Header(default=None)) -> dict:
        _auth(x_internal_token)
        from .trace import TraceStore

        store = TraceStore()
        with store._conn() as c:
            rows = c.execute(
                """SELECT id, ts, question, route, grounded, degraded, rewrites,
                          latency_ms, prompt_tokens, completion_tokens, n_citations, answer
                   FROM traces ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return {"items": [dict(r) for r in rows]}

    return app


def serve(host: str | None = None, port: int | None = None) -> None:
    """启动服务。走 uvicorn。"""
    if not _HAS_FASTAPI:  # pragma: no cover
        raise SystemExit(
            "缺少 web 依赖，无法启动服务。请先安装：\n"
            "  pip install 'citepilot[server]'\n"
            "或者临时装： pip install fastapi uvicorn"
        )
    import uvicorn

    h = host or os.getenv("CITEPILOT_SERVE_HOST", "127.0.0.1")
    p = int(port or os.getenv("CITEPILOT_SERVE_PORT", "8077"))

    print(f"CitePilot 内核启动中 → http://{h}:{p}")
    print(f"  接口文档   http://{h}:{p}/docs")
    print(f"  健康检查   http://{h}:{p}/health")
    tok = os.getenv("CITEPILOT_INTERNAL_TOKEN", "").strip()
    print(f"  内部 token {'已启用' if tok else '未设置（仅限本机调试）'}")
    uvicorn.run(create_app(), host=h, port=p, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    serve()
