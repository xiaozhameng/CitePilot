"""切分公共设施：长度策略、短块合并、通用打包器。

为什么要把这三样单独抽出来？
因为现在有两个切分器：
  · ingest.py —— 支付手册的 HTML（贴着它自己的 DOM 结构走）
  · kb.py     —— 上传的文档（md / txt / html / pdf / docx 五种）
它们对「一个块多长合适」「多短算碎片」的判断必须**完全一致**。
不一致的后果不是报错，而是很隐蔽的偏差：同一类问题在手册上召回到 3 条、
在上传文档上只召回 1 条，然后你会去怀疑向量模型、怀疑分词器，
真实原因只是两份代码里的 MIN_CHARS 差了几十个字符。

这和 RetrieveLoop 那次是同一类教训：**控制逻辑只留一份**。

分工：
  · 长度常量 + norm_text + merge_short  → 纯函数，两个切分器共用（本文件）
  · BlockPacker                        → 「标题 + 正文」流式输入打包器，给 kb.py 用
  · 手册的 DOM 走读逻辑                 → 留在 ingest.py（那是它独有的）
"""

from __future__ import annotations

import re

from .schema import Chunk

# 单个 Chunk 的目标长度（字符）。太小丢上下文，太大稀释向量语义。
MAX_CHARS = 480

# 低于这个长度的碎片会被并入相邻块。整章导语、图注这类短句单独成块时，
# 会把 Top-K 的名额占掉，而它们本身没有足够信息回答任何问题。
MIN_CHARS = 60


def norm_text(s: str) -> str:
    """归一化空白：HTML / PDF 抽出来的换行缩进全是噪音。"""
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()


def merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """把过短的碎片并入相邻块。

    为什么必须做：检索是按块算分的，一个 4 个字的块（"读完之后"）如果
    因为某种词面巧合排进了 Top-5，就会挤掉一个真正有内容的块，
    而它作为"出处"毫无价值 —— 引用一段 4 个字等于没有出处。
    合并只在同一章内进行，且不跨越问答块（问答块有自己的结构）。
    """
    out: list[Chunk] = []
    buf: Chunk | None = None

    def push(c: Chunk) -> None:
        """把一个（可能过短的）块送进结果，过短就并入前一个块。"""
        if len(c.text) < MIN_CHARS and out and out[-1].kind == "prose":
            out[-1].text = f"{out[-1].text}\n{c.text}"
        else:
            out.append(c)

    for c in chunks:
        if c.kind != "prose":
            # 问答块有自己的结构，不能和散文混在一起。
            # 但缓冲区里攥着的那块不能直接扔掉条件判断 —— 这里是第一版的漏洞：
            # 碰到问答块就直接 append，短块就这么漏进了语料（实测出现 51 字的块）。
            if buf is not None:
                push(buf)
                buf = None
            out.append(c)
            continue
        if buf is None:
            buf = c
            continue
        if len(buf.text) < MIN_CHARS:
            # 累积：两个 25 字的碎片并起来还是 50 字，仍然不够，得继续攒
            c.text = f"{buf.text}\n{c.text}"
            buf = c
        else:
            out.append(buf)
            buf = c
    if buf is not None:
        push(buf)

    # 兜底清理：上面两轮只能处理"相邻"的情况，仍然会漏掉两类块 ——
    #   1) 紧跟问答块之后的短散文（前面的块不是散文，push 没地方并）
    #   2) 一章里排在很前面的短块（前面还没有散文块可并）
    # 这两类如果不处理，就会出现 4 个字的"出处"（实测真有）。
    i = 0
    while i < len(out):
        c = out[i]
        if c.kind == "prose" and len(c.text) < MIN_CHARS and len(out) > 1:
            nxt = next((j for j in range(i + 1, len(out)) if out[j].kind == "prose"), None)
            prv = next((j for j in range(i - 1, -1, -1) if out[j].kind == "prose"), None)
            if nxt is not None:
                out[nxt].text = f"{c.text}\n{out[nxt].text}"
                del out[i]
                continue
            if prv is not None:
                out[prv].text = f"{out[prv].text}\n{c.text}"
                del out[i]
                continue
        i += 1
    # 重新编号：合并之后原来的序号会出现空洞，且块数变了。
    # 编号只求唯一和稳定，所以用「章 id + 序号」即可。
    for i, c in enumerate(out, 1):
        c.chunk_id = f"{c.chapter_id}::c{i:03d}"
    return out


class BlockPacker:
    """把「标题 + 正文」的块序列打包成 Chunk。

    给上传的文档用。五种格式（md / txt / html / pdf / docx）抽出来的文本
    最终都是同一个形状：一串 (level, title, body)。统一在这里做累积、
    切分、短块合并 —— 否则五种格式要各写一遍打包逻辑，迟早有一份走偏。

    level 的约定：0 = 正文块（只有 body），1 = 一级标题，2+ = 子标题。

    为什么标题天然是块的边界？
    因为「引用要能指到哪一节」。如果让一段文字横跨两个小节，
    引用标签就只能模糊到"这附近"，等于把出处做废了。
    """

    def __init__(self, *, doc_id: str, doc_name: str) -> None:
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.chunks: list[Chunk] = []
        self._buf: list[str] = []
        self._top = ""        # 最近的一级标题
        self._section = ""    # 最近的小节标题
        self._sec_idx = 0
        self._seq = 0

    def heading(self, level: int, title: str) -> None:
        self.flush()
        title = norm_text(title)
        if not title:
            return
        if level <= 1:
            self._top = title
            self._section = ""
        else:
            self._section = title
        self._sec_idx += 1

    def body(self, text: str) -> None:
        text = norm_text(text)
        if not text:
            return
        self._buf.append(text)
        if sum(len(x) for x in self._buf) >= MAX_CHARS:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        body = "\n".join(self._buf)
        self._buf = []
        self._seq += 1
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.doc_id}::s{self._sec_idx:03d}::{self._seq:03d}",
                chapter_id=f"{self.doc_id}::s{self._sec_idx:03d}",
                chapter_no="",
                chapter_title=self._top,
                part="",
                section_title=self._section,
                text=body,
                kind="prose",
                order=0,
                doc_id=self.doc_id,
                doc_name=self.doc_name,
                origin="upload",
            )
        )

    def result(self) -> list[Chunk]:
        self.flush()
        out = merge_short(self.chunks)
        for i, c in enumerate(out, 1):
            c.order = i
        return out
