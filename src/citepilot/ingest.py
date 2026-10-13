"""语料管线：把电子书的 HTML 切成 Chunk。

为什么不用现成的文档加载器（LangChain 的 HTMLHeaderTextSplitter 之类）？
因为「出处」这件事需要章节级的元数据。通用的按字符数滑窗会把一节切碎、
丢掉所属章节，引用就只能显示「第 3 段」这种没有信息量的东西。
这里贴着真实 HTML 结构走：章（section.chapter）→ 节（h2）→ 段落，
所以每个 Chunk 都自带「第几章 · 第几节」，引用可以直接跳到原文锚点。

一条重要的方法论：自测题章节（ch-appx-quiz）**不进索引**。
它是评测集的答案，如果索引了，检索就等于开卷考试，指标全是假的。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .schema import Chunk
# 长度策略、短块合并、空白归一化都放在 split.py —— 上传文档的切分器要跟这里
# 用同一套标准，所以不能留在本文件里各写一份。别名保留了原来的私有名，
# 是为了让下面的走读代码一个字都不用改（改名的批量替换最容易误伤 _norm_chapter_no）。
from .split import (
    MAX_CHARS,
    MIN_CHARS,
    merge_short as _merge_short,
    norm_text as _norm,
)

# 这些标签里的东西不是知识，是渲染物
SKIP_TAGS = {"script", "style", "svg", "defs", "nav", "aside", "noscript", "button"}

# 这些标签本身就是一段正文
BLOCK_TAGS = {"p", "li", "figcaption", "summary", "blockquote", "dt", "dd"}

# 不参与索引的章节：自测题是评测集的答案，必须排除
EXCLUDE_FROM_INDEX = {"ch-appx-quiz"}


def _norm_chapter_no(raw: str) -> str:
    """把 'PART 3 · CH 8' 归一化成 '第 8 章'。

    为什么要做这一步？
    因为引用要给人看。'PART 3 · CH 8 银联 · 数字人民币 · 网联 · 8.1 四个角色的定位'
    又长又难扫；'第 8 章 银联 · 数字人民币 · 网联 · 8.1 四个角色的定位' 一眼就能定位。
    同时归一化出来的数字还能直接给评测集做章号→chapter_id 的映射。
    """
    raw = raw or ""
    m = re.search(r"\bCH\s*(\d+)\b", raw)
    if m:
        return f"第 {int(m.group(1))} 章"
    m = re.search(r"APPENDIX\s*([A-Z])", raw, re.I)
    if m:
        return f"附录 {m.group(1).upper()}"
    if "FAQ" in raw.upper():
        return "FAQ 速查"
    if "GUIDE" in raw.upper():
        return "导读"
    return raw.strip()


def _txt(node: Tag) -> str:
    return _norm(node.get_text(" ", strip=True))


def _table_md(tbl: Tag) -> str:
    """表格转成竖线文本。表格里往往就是最密集的知识点，不能丢。"""
    rows: list[str] = []
    for tr in tbl.find_all("tr"):
        cells = [_txt(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


@dataclass
class _Ctx:
    chapter_id: str
    chapter_no: str
    chapter_title: str
    part: str
    section: str = ""
    sub: str = ""
    buf: list[str] = None          # type: ignore[assignment]
    chunks: list[Chunk] = None     # type: ignore[assignment]
    seq: int = 0
    order: int = 0
    sec_idx: int = 0

    def __post_init__(self) -> None:
        self.buf = []
        self.chunks = []

    def add(self, text: str, kind: str = "prose") -> None:
        text = _norm(text)
        if not text:
            return
        self.buf.append(text)
        if sum(len(x) for x in self.buf) >= MAX_CHARS:
            self.flush(kind)

    def add_qa(self, q: str, a: str) -> None:
        """问答对单独成块 —— 它的结构（问题+答案）比正文更适合被问句命中。"""
        self.flush()
        body = f"问：{_norm(q)}\n答：{_norm(a)}"
        self._emit(body, "qa")

    def flush(self, kind: str = "prose") -> None:
        if not self.buf:
            return
        self._emit("\n".join(self.buf), kind)
        self.buf = []

    def _emit(self, body: str, kind: str) -> None:
        self.seq += 1
        self.order += 1
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.chapter_id}::s{self.sec_idx}::{self.seq}",
                chapter_id=self.chapter_id,
                chapter_no=self.chapter_no,
                chapter_title=self.chapter_title,
                part=self.part,
                section_title=self.sub or self.section,
                text=body,
                kind=kind,
                order=self.order,
            )
        )


def _walk(node: Tag, ctx: _Ctx) -> None:
    """按文档顺序下钻，遇到块级元素就收集文本。"""
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in SKIP_TAGS:
            continue

        if name == "h2":
            ctx.flush()
            ctx.section = _txt(child)
            ctx.sub = ""
            ctx.sec_idx += 1
            continue
        if name in ("h3", "h4"):
            ctx.flush()
            ctx.sub = _txt(child)
            continue
        if name == "table":
            ctx.add(_table_md(child))
            continue

        classes = set(child.get("class") or [])

        if "ch-head" in classes:
            # 章头（章号 / 章标题 / 导语）在外面已经处理过了。
            # 这里不跳过的话，导语会被收两遍 —— 第一版就是这么错的，
            # 结果语料里出现成对的重复段落。
            continue
        if "fig-cap" in classes:
            # 图注往往就是全章的结论，是密度最高的文本
            ctx.add(_txt(child), "figure_caption")
            continue
        if "qa" in classes:
            q, a = child.find("div", class_="q"), child.find("div", class_="a")
            if q and a:
                ctx.add_qa(_txt(q), _txt(a))
            continue
        if name == "details" and "q" in classes:
            s, a = child.find("summary"), child.find("div", class_="ans")
            if s and a:
                ctx.add_qa(_txt(s), _txt(a))
            continue
        if name in BLOCK_TAGS:
            ctx.add(_txt(child))
            continue
        if name == "span" and "t" in classes:
            # 例 / 卡片的标题行
            ctx.add(_txt(child))
            continue

        # 其余是容器（div.card / div.example / ul / section…），继续下钻
        _walk(child, ctx)


def parse_chapters(html_path: Path) -> list[dict[str, str]]:
    """只抽章节元信息，供目录与引用标签使用。"""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    out: list[dict[str, str]] = []
    for sec in soup.find_all("section", class_="chapter"):
        head = sec.find("div", class_="ch-head")
        if not head:
            continue
        h1 = head.find("h1")
        num = head.find("div", class_="ch-num")
        tag = head.find("span", class_="part-tag")
        out.append(
            {
                "chapter_id": sec.get("id", ""),
                "chapter_no": _txt(num) if num else "",
                "chapter_title": _txt(h1) if h1 else "",
                "part": _txt(tag) if tag else "",
            }
        )
    return out


def build_chunks(html_path: Path, exclude: set[str] | None = None) -> list[Chunk]:
    exclude = EXCLUDE_FROM_INDEX if exclude is None else exclude
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    chunks: list[Chunk] = []

    for sec in soup.find_all("section", class_="chapter"):
        cid = sec.get("id", "")
        if not cid or cid in exclude:
            continue
        head = sec.find("div", class_="ch-head")
        ctx = _Ctx(
            chapter_id=cid,
            chapter_no=_norm_chapter_no(_txt(head.find("div", class_="ch-num"))) if head and head.find("div", class_="ch-num") else "",
            chapter_title=_txt(head.find("h1")) if head and head.find("h1") else "",
            part=_txt(head.find("span", class_="part-tag")) if head and head.find("span", class_="part-tag") else "",
        )
        # 导读单独成块：它是全章的摘要，很多"这本书讲了什么"类问题的答案在这里
        if head:
            lede = head.find("p", class_="lede")
            if lede:
                ctx.buf.append(_norm(_txt(lede)))
                ctx.flush()
            _walk(sec, ctx)
        else:
            _walk(sec, ctx)
        ctx.flush()
        chunks.extend(_merge_short(ctx.chunks))

    for i, c in enumerate(chunks, 1):
        c.order = i
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Chunk.from_dict(json.loads(line)))
    return out


def extract_quiz(html_path: Path) -> list[dict[str, str]]:
    """抽自测题（20 题）。每题原文都标了出处章节，是现成的检索标注。"""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    out: list[dict[str, str]] = []
    for sec in soup.find_all("section", class_="chapter"):
        if sec.get("id") != "ch-appx-quiz":
            continue
        for d in sec.find_all("details", class_="q"):
            s, a = d.find("summary"), d.find("div", class_="ans")
            if s and a:
                out.append({"question": _txt(s), "answer": _txt(a)})
    return out


def extract_glossary(html_path: Path) -> list[dict[str, str]]:
    """抽术语表。术语名天然就是一条查询（"XX 是什么"），可直接当评测样本。"""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    out: list[dict[str, str]] = []
    for sec in soup.find_all("section", class_="chapter"):
        if sec.get("id") != "ch-appx-glossary":
            continue
        for p in sec.find_all("p", class_="term"):
            w, d = p.find("span", class_="w"), p.find("span", class_="d")
            if w and d:
                out.append({"term": _txt(w), "definition": _txt(d)})
    return out


if __name__ == "__main__":
    from .config import SETTINGS

    src = SETTINGS.corpus_dir / "index.html"
    cs = build_chunks(src)
    dest = SETTINGS.corpus_dir / "chunks.jsonl"
    save_chunks(cs, dest)
    quiz, glos = extract_quiz(src), extract_glossary(src)
    print(f"切出 {len(cs)} 个 Chunk → {dest}")
    print(f"自测题 {len(quiz)} 条、术语 {len(glos)} 条（用于评测集，不进索引）")
    kinds: dict[str, int] = {}
    for c in cs:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    print("类型分布:", kinds)
    avg = sum(len(c.text) for c in cs) / max(len(cs), 1)
    print(f"平均长度 {avg:.0f} 字符，最长 {max(len(c.text) for c in cs)}，最短 {min(len(c.text) for c in cs)}")
