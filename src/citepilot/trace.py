"""可观测性：把每次问答的链路落到 SQLite。

为什么必须做这一层？
因为「答非所问」「P95 太高」「成本超了」这三类问题，
靠看日志是查不出来的 —— 你需要能按维度聚合的结构化数据：
  · 哪一步最慢？      → steps 表按 name 聚合 ms
  · 钱花在哪？        → token 按 prompt/completion 拆开，看是不是上下文太长
  · 降级了多少次？    → degraded 字段的取值分布（这是"降级链"是否真的生效的证据）
  · 引用可信度如何？  → grounded 为 False 的比例

这些数字就是实践里"你的系统 P95 多少、单次成本多少"的答案来源。
没有这一层，那些数字只能靠估 —— 一被追问就露馅。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .config import SETTINGS
from .schema import Answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    question     TEXT    NOT NULL,
    route        TEXT,
    intent       TEXT,
    confidence   REAL,
    rewrites     INTEGER,
    grounded     INTEGER,
    degraded     TEXT,
    latency_ms   INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    n_hits       INTEGER,
    n_citations  INTEGER,
    top_chunk    TEXT,
    answer       TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id     INTEGER NOT NULL,
    name         TEXT    NOT NULL,
    ms           INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    meta         TEXT,
    FOREIGN KEY(trace_id) REFERENCES traces(id)
);
CREATE INDEX IF NOT EXISTS idx_steps_trace ON steps(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_ts   ON traces(ts);
"""


class TraceStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or SETTINGS.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, ans: Answer, model: str = "") -> int:
        total_ms = ans.latency_ms  # 已经排除了 total 步骤，不会重复累加
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO traces (ts, question, route, intent, confidence, rewrites,
                       grounded, degraded, latency_ms, prompt_tokens, completion_tokens,
                       n_hits, n_citations, top_chunk, answer)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(), ans.question, ans.route, ans.intent, ans.intent_confidence,
                    ans.rewrites, 1 if ans.grounded else 0, ans.degraded, total_ms,
                    ans.prompt_tokens, ans.completion_tokens, len(ans.hits),
                    len(ans.citations), ans.hits[0].chunk_id if ans.hits else "", ans.text,
                ),
            )
            trace_id = int(cur.lastrowid)
            c.executemany(
                """INSERT INTO steps (trace_id, name, ms, prompt_tokens, completion_tokens, meta)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (trace_id, s.name, s.ms, s.prompt_tokens, s.completion_tokens,
                     json.dumps(s.meta, ensure_ascii=False))
                    for s in ans.steps
                ],
            )
        return trace_id

    # ---------- 聚合查询：直接出评审要说的数字 ----------
    def summary(self) -> dict:
        with self._conn() as c:
            row = c.execute(
                """SELECT COUNT(*) n,
                          AVG(latency_ms) avg_ms,
                          MAX(latency_ms) max_ms,
                          SUM(prompt_tokens) pin,
                          SUM(completion_tokens) pout,
                          SUM(CASE WHEN grounded = 0 THEN 1 ELSE 0 END) ungrounded,
                          SUM(CASE WHEN degraded <> '' THEN 1 ELSE 0 END) degraded_n,
                          SUM(rewrites) rewrites
                   FROM traces"""
            ).fetchone()
            steps = c.execute(
                """SELECT name, COUNT(*) n, AVG(ms) avg_ms, MAX(ms) max_ms,
                          SUM(prompt_tokens + completion_tokens) tokens
                   FROM steps GROUP BY name ORDER BY avg_ms DESC"""
            ).fetchall()
            deg = c.execute(
                """SELECT degraded, COUNT(*) n FROM traces
                   WHERE degraded <> '' GROUP BY degraded ORDER BY n DESC"""
            ).fetchall()
        return {
            "total": row["n"] or 0,
            "avg_ms": row["avg_ms"] or 0,
            "max_ms": row["max_ms"] or 0,
            "prompt_tokens": row["pin"] or 0,
            "completion_tokens": row["pout"] or 0,
            "ungrounded": row["ungrounded"] or 0,
            "degraded_n": row["degraded_n"] or 0,
            "rewrites": row["rewrites"] or 0,
            "steps": [dict(s) for s in steps],
            "degraded_breakdown": [dict(d) for d in deg],
        }

    def percentile(self, p: float) -> int:
        """延迟分位数。样本少的时候分位数意义有限，但趋势能看。"""
        with self._conn() as c:
            rows = [r[0] for r in c.execute("SELECT latency_ms FROM traces ORDER BY latency_ms")]
        if not rows:
            return 0
        k = max(0, min(len(rows) - 1, int(round(p / 100 * len(rows) + 0.5)) - 1))
        return int(rows[k])
