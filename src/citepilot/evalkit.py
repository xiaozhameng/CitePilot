"""评测工具箱：评测集构建 + 检索指标。

评测集从哪来？——不从天上掉，也不靠我编。
本书自带两处**作者本人标注过出处**的内容：

  1) 附录 C「架构师自测 20 题」：每题答案末尾都写了「见第 3 章」这样的出处。
     这是天然的 (question → 正确章节) 标注，20 条。
  2) 附录 A「支付黑话词典」：术语名本身就是一条检索 query，
     正确来源就是收录该术语的那个 Chunk。

用作者自己的标注当 ground truth，比我自己造问题可信得多 ——
而且它可复现：谁跑都是同 20 条，改了切分策略就能直接对比。
（附录 C 已被排除出索引，否则等于开卷考试，指标全假。）

指标口径：
  章级 Recall@K —— 前 K 条里命中了正确章节的比例（主指标，对切分策略不敏感）
  章级 MRR     —— 第一条正确结果的排名倒数（衡量"排得够不够前"）
  NDCG@K       —— 考虑位置和等级的排序质量
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import extract_glossary, extract_quiz
from .schema import Chunk

_CHAPTER_REF = re.compile(r"第\s*((?:\d+\s*[、,，]?\s*)+)章")


def _chapter_numbers(text: str) -> list[int]:
    """从「见第 0、2 章」里抠出 [0, 2]。"""
    nums: list[int] = []
    for m in _CHAPTER_REF.finditer(text):
        for n in re.findall(r"\d+", m.group(1)):
            v = int(n)
            if v not in nums:
                nums.append(v)
    return nums


def _chapter_no_map(chunks: list[Chunk]) -> dict[int, str]:
    """章号数字 → chapter_id，例如 3 → ch-03。"""
    out: dict[int, str] = {}
    for c in chunks:
        digits = re.findall(r"\d+", c.chapter_no or "")
        if digits:
            n = int(digits[0])
            out.setdefault(n, c.chapter_id)
    return out


@dataclass
class GoldItem:
    qid: str
    question: str
    source: str                    # quiz | glossary
    gold_chapters: list[str] = field(default_factory=list)   # 期望命中的 chapter_id
    gold_chunk: str = ""           # 期望命中的 chunk_id（术语题用）
    reference: str = ""            # 参考答案，供人工核对与忠实度评估

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "question": self.question, "source": self.source,
            "gold_chapters": self.gold_chapters, "gold_chunk": self.gold_chunk,
            "reference": self.reference,
        }


def build_goldenset(html_path: Path, chunks: list[Chunk]) -> list[GoldItem]:
    no_map = _chapter_no_map(chunks)
    items: list[GoldItem] = []

    # ---- 来源一：自测题 20 条 ----
    for i, q in enumerate(extract_quiz(html_path), 1):
        ch_ids = [no_map[n] for n in _chapter_numbers(q["answer"]) if n in no_map]
        if not ch_ids:
            continue  # 没标出处的题不用（比如"这本书怎么用"类）
        items.append(GoldItem(
            qid=f"quiz-{i:02d}", question=q["question"], source="quiz",
            gold_chapters=ch_ids, reference=q["answer"],
        ))

    # ---- 来源二：术语表 ----
    # 术语的定义就在词典章里，所以正确来源 = 收录该术语的那个 Chunk
    for i, g in enumerate(extract_glossary(html_path), 1):
        term = g["term"].strip()
        hit = _find_term_chunk(term, chunks)
        if not hit:
            continue
        ch_ids = [no_map[n] for n in _chapter_numbers(g["definition"]) if n in no_map]
        items.append(GoldItem(
            qid=f"glossary-{i:02d}", question=f"{term}是什么？", source="glossary",
            gold_chapters=sorted(set(ch_ids + [hit.chapter_id])),
            gold_chunk=hit.chunk_id, reference=g["definition"],
        ))
    return items


def _find_term_chunk(term: str, chunks: list[Chunk]) -> Chunk | None:
    """定位收录某术语定义的那个 Chunk。

    不能只认「术语在块首」—— 合并短块之后，一个块里常常连着收录好几个术语，
    只有第一个才在块首，剩下的全会被漏掉（第一版就是这样，39 条只认出 9 条）。
    改成「块里出现过这个词」，并按三个条件排序取最优：
      1) 优先词典章（ch-appx-glossary）—— 那里才是定义本身
      2) 术语出现的位置越靠前越好
      3) 同样靠前时取更短的块（更聚焦）
    """
    key = "".join(term.split())
    best: Chunk | None = None
    best_score: tuple[int, int, int] | None = None
    for c in chunks:
        if c.kind == "figure_caption":
            continue
        body = "".join(c.text.split())
        pos = body.find(key)
        if pos < 0:
            continue
        score = (0 if c.chapter_id == "ch-appx-glossary" else 1, pos, len(c.text))
        if best_score is None or score < best_score:
            best, best_score = c, score
    return best


# ---------------- 指标 ----------------
def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def recall_at_k(hits: list[list[str]], golds: list[list[str]], k: int) -> float:
    """章级 Recall@K：前 K 条里是否命中了正确章节。

    这里用的是"命中即算"的口径（binary recall），而不是"覆盖了全部正确章节的比例" ——
    因为自测题常常标注了 2~3 个章节，要求全部覆盖会让指标虚低，
    对"能不能找到答案"这个真实目标也不公平。
    """
    if not hits:
        return 0.0
    ok = 0
    for h, g in zip(hits, golds):
        if not g:
            continue
        if set(h[:k]) & set(g):
            ok += 1
    return ok / max(len(hits), 1)


def mrr(hits: list[list[str]], golds: list[list[str]], k: int = 100) -> float:
    total = 0.0
    n = 0
    for h, g in zip(hits, golds):
        if not g:
            continue
        n += 1
        for rank, cid in enumerate(h[:k], 1):
            if cid in g:
                total += 1.0 / rank
                break
    return total / max(n, 1)


def ndcg_at_k(hits: list[list[str]], golds: list[list[str]], k: int) -> float:
    total = 0.0
    n = 0
    for h, g in zip(hits, golds):
        if not g:
            continue
        n += 1
        rels = [1 if cid in g else 0 for cid in h[:k]]
        ideal = sorted(rels, reverse=True)
        idcg = _dcg(ideal)
        total += (_dcg(rels) / idcg) if idcg else 0.0
    return total / max(n, 1)


def load_goldenset(path: Path) -> list[GoldItem]:
    fields = set(GoldItem.__dataclass_fields__)
    out: list[GoldItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # 容忍多余字段：口语化难样本会多带一个 original 原始问句，
            # 缺字段补默认值。评测集的格式不该因为加一个备注就崩。
            out.append(GoldItem(**{k: v for k, v in d.items() if k in fields}))
    return out
