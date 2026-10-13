"""语料切分的测试。

切分是最容易被忽略、但影响最大的一层：
它错了，后面所有指标都建立在错误的语料上，而且很难察觉。
第一版就出过两个具体的错，这两个测试就是为了把它们钉住。
"""

from pathlib import Path

import pytest

from citepilot.config import SETTINGS
from citepilot.ingest import (
    EXCLUDE_FROM_INDEX,
    MIN_CHARS,
    build_chunks,
    extract_glossary,
    extract_quiz,
    load_chunks,
    save_chunks,
    _norm_chapter_no,
)

CORPUS = SETTINGS.corpus_dir / "index.html"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="需要语料文件")


def test_chapter_no_is_normalized_for_citation():
    """'PART 3 · CH 8' 这种内部标记不能出现在引用里，要变成'第 8 章'。"""
    assert _norm_chapter_no("PART 3 · CH 8") == "第 8 章"
    assert _norm_chapter_no("CH 00 · 全景") == "第 0 章"
    assert _norm_chapter_no("APPENDIX A") == "附录 A"
    assert _norm_chapter_no("PART 8 · FAQ") == "FAQ 速查"
    assert _norm_chapter_no("GUIDE") == "导读"


def test_quiz_chapter_is_excluded_from_index():
    """自测题章节是评测集的答案，绝不能进索引，否则评测就是开卷考试。"""
    chunks = build_chunks(CORPUS)
    assert "ch-appx-quiz" in EXCLUDE_FROM_INDEX
    assert all(c.chapter_id != "ch-appx-quiz" for c in chunks)


def test_no_duplicate_chunks():
    """章导语曾被收录两遍（ch-head 被遍历了两次），导致语料里出现成对重复段落。"""
    chunks = build_chunks(CORPUS)
    texts = ["".join(c.text.split()) for c in chunks]
    assert len(texts) == len(set(texts)), "存在完全重复的 Chunk"


def test_no_ultra_short_chunks():
    """保留 4 个字的碎片当"出处"没有意义，必须并入相邻块。"""
    chunks = build_chunks(CORPUS)
    assert min(len(c.text) for c in chunks) >= MIN_CHARS


def test_chunk_ids_are_unique_and_ordered():
    chunks = build_chunks(CORPUS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.order for c in chunks] == sorted(c.order for c in chunks)


def test_every_chunk_can_build_a_citation_label_and_anchor():
    """每个块都必须能拼出「出处标签」和「可跳转锚点」，这是需求的地基。"""
    for c in build_chunks(CORPUS):
        assert c.chapter_no and c.chapter_title
        assert c.label.startswith(c.chapter_no)
        assert c.anchor.endswith("#" + c.chapter_id)


def test_quiz_and_glossary_are_extractable():
    quiz = extract_quiz(CORPUS)
    glos = extract_glossary(CORPUS)
    assert len(quiz) == 20
    assert len(glos) >= 30
    # 自测题的答案里带出处标注，这是评测集的标注来源
    assert any("见第" in q["answer"] for q in quiz)


def test_roundtrip_save_load(tmp_path: Path):
    chunks = build_chunks(CORPUS)
    p = tmp_path / "chunks.jsonl"
    save_chunks(chunks, p)
    back = load_chunks(p)
    assert len(back) == len(chunks)
    assert back[0].text == chunks[0].text
    assert back[0].label == chunks[0].label
