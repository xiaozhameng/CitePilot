"""知识库（手动上传）的测试。

重点覆盖三类最容易出错的地方：
  1) 没有标题结构的文档（纯文本 / 扫描版 PDF）是否还能给出有用的出处，或明确报错
  2) doc_id 的生成与冲突（中文名、路径分隔符）
  3) 解析出 0 块时必须报错 —— 否则库里会留下「显示已上传、检索永远召不回」的幽灵记录
"""

from __future__ import annotations

import pytest

from citepilot.kb import KnowledgeBase, extract_blocks, make_doc_id
from citepilot.schema import Chunk
from citepilot.split import merge_short, norm_text


# ------------------------------------------------------------------
# 出处标签与深链：内置语料必须保持原样，上传文档必须带文档名
# ------------------------------------------------------------------
def test_label_and_anchor_for_upload() -> None:
    c = Chunk(
        chunk_id="我的文档::s001::c001", chapter_id="我的文档::s001", chapter_no="",
        chapter_title="", part="", section_title="同城双活",
        text="正文", doc_id="我的文档", doc_name="我的文档", origin="upload",
    )
    assert c.label == "我的文档 · 同城双活"
    assert c.anchor.startswith("/v1/kb/documents/")
    assert "/raw?chunk=" in c.anchor


def test_label_and_anchor_for_upload_without_section() -> None:
    """纯文本没有小节，出处只能是文档级 —— 但不能变成空字符串。"""
    c = Chunk(
        chunk_id="d::s000::c001", chapter_id="d::s000", chapter_no="", chapter_title="",
        part="", section_title="", text="正文", doc_id="d", doc_name="说明文档",
        origin="upload",
    )
    assert c.label == "说明文档"


def test_label_and_anchor_for_builtin_unchanged() -> None:
    """内置语料的标签格式一个字都不能变：57 条评测集和既有引用都依赖它。"""
    c = Chunk(
        chunk_id="ch-03::c001", chapter_id="ch-03", chapter_no="第 3 章",
        chapter_title="支付体系", part="", section_title="3.1 概览", text="正文",
    )
    assert c.label == "第 3 章 支付体系 · 3.1 概览"
    assert c.anchor == "corpus/payment-handbook/index.html#ch-03"


def test_upload_anchor_is_url_encoded() -> None:
    """doc_id 允许中文，所以锚点必须做百分号编码，否则前端点不开。"""
    c = Chunk(
        chunk_id="中文文档::s001::c001", chapter_id="中文文档::s001", chapter_no="",
        chapter_title="", part="", section_title="节", text="x",
        doc_id="中文文档", doc_name="中文文档", origin="upload",
    )
    assert "%" in c.anchor
    assert " " not in c.anchor


# ------------------------------------------------------------------
# doc_id 生成
# ------------------------------------------------------------------
def test_make_doc_id_keeps_readable_name() -> None:
    assert make_doc_id("容灾说明.md", set()) == "容灾说明"


def test_make_doc_id_dedupes() -> None:
    assert make_doc_id("报告.md", {"报告"}) == "报告-2"
    assert make_doc_id("报告.md", {"报告", "报告-2"}) == "报告-3"


def test_make_doc_id_strips_dangerous_chars() -> None:
    got = make_doc_id('a:b*c?d.md', set())
    assert not set(got) & set('/\\:*?"<>|')


def test_make_doc_id_never_returns_empty() -> None:
    """doc_id 绝不能为空 —— 空 id 会让文档目录等于知识库根目录，直接把整个库搞坏。"""
    for raw in (".md", "///.md", "   .md", "...md", "??.md"):
        got = make_doc_id(raw, set())
        assert got, f"{raw!r} 生成了空 id"
        assert not set(got) & set('/\\:*?"<>|')


# ------------------------------------------------------------------
# 各格式的文本抽取
# ------------------------------------------------------------------
def test_markdown_headings_become_levels() -> None:
    data = "# 总标题\n\n引言。\n\n## 小节一\n\n正文一。\n\n## 小节二\n\n正文二。\n".encode()
    blocks = list(extract_blocks(".md", data))
    heads = [(lvl, title) for lvl, title, _ in blocks if lvl > 0]
    assert heads == [(1, "总标题"), (2, "小节一"), (2, "小节二")]


def test_markdown_hash_inside_code_fence_is_not_heading() -> None:
    """代码块里的 # 是注释，不是标题 —— 认错了会把代码注释当成章节名。"""
    data = "## 真标题\n\n```sh\n# 这是注释不是标题\necho hi\n```\n\n正文。\n".encode()
    blocks = list(extract_blocks(".md", data))
    heads = [title for lvl, title, _ in blocks if lvl > 0]
    assert heads == ["真标题"]


def test_plain_text_yields_no_headings() -> None:
    blocks = list(extract_blocks(".txt", "第一段。\n\n第二段。".encode()))
    assert all(lvl == 0 for lvl, _, _ in blocks)
    assert len(blocks) == 2


def test_gbk_encoded_text_is_decoded() -> None:
    """国内文档常见 GBK 编码，只试 utf-8 会整份失败。"""
    blocks = list(extract_blocks(".txt", "中文段落内容。".encode("gb18030")))
    assert any("中文段落" in body for _, _, body in blocks)


def test_unknown_suffix_is_rejected() -> None:
    with pytest.raises(ValueError, match="不支持"):
        list(extract_blocks(".xlsx", b"x"))


# ------------------------------------------------------------------
# 入库 / 出库流程
# ------------------------------------------------------------------
def test_add_markdown_and_assign_sections(tmp_path) -> None:
    kb = KnowledgeBase(tmp_path / "kb")
    md = "# 总标题\n\n## 第一节\n\n" + "甲" * 80 + "\n\n## 第二节\n\n" + "乙" * 80 + "\n"
    info = kb.add_document("测试.md", md.encode("utf-8"))

    assert info.chunk_count == 2
    assert info.suffix == ".md"
    chunks = kb.load_chunks()
    assert [c.section_title for c in chunks] == ["第一节", "第二节"]
    assert all(c.origin == "upload" for c in chunks)
    assert all(c.doc_id == info.doc_id for c in chunks)
    assert all(c.doc_name == "测试" for c in chunks)


def test_add_document_keeps_original_file(tmp_path) -> None:
    """原件要留着：切分策略改了以后能重跑，不用让用户重新上传。"""
    kb = KnowledgeBase(tmp_path / "kb")
    info = kb.add_document("a.md", ("# t\n\n" + "正文" * 40).encode())
    original = kb.root / info.doc_id / "original.md"
    assert original.exists()
    assert original.read_bytes().startswith("# t".encode())


def test_add_duplicate_filename_gets_new_id(tmp_path) -> None:
    kb = KnowledgeBase(tmp_path / "kb")
    a = kb.add_document("同名.md", ("# A\n\n" + "甲" * 80).encode())
    b = kb.add_document("同名.md", ("# B\n\n" + "乙" * 80).encode())
    assert a.doc_id != b.doc_id
    assert len(kb.list_documents()) == 2
    assert len(kb.load_chunks()) == 2


def test_remove_document(tmp_path) -> None:
    kb = KnowledgeBase(tmp_path / "kb")
    info = kb.add_document("a.md", ("# t\n\n" + "正文" * 40).encode())
    assert kb.remove_document(info.doc_id) is True
    assert kb.list_documents() == []
    assert kb.load_chunks() == []
    assert kb.remove_document(info.doc_id) is False
    assert not (kb.root / info.doc_id).exists()


def test_empty_file_is_rejected(tmp_path) -> None:
    kb = KnowledgeBase(tmp_path / "kb")
    with pytest.raises(ValueError, match="空"):
        kb.add_document("empty.md", b"")


def test_whitespace_only_document_is_rejected(tmp_path) -> None:
    """空白文档必须报错。悄悄入库的话，这条记录永远召不回任何东西。"""
    kb = KnowledgeBase(tmp_path / "kb")
    with pytest.raises(ValueError, match="没有切出任何文字"):
        kb.add_document("blank.md", b"   \n\n   \n")
    assert kb.list_documents() == []


def test_scanned_pdf_without_text_layer_is_rejected(tmp_path) -> None:
    """扫描版 PDF 没有文字层 —— 这是最常见的上传失败，必须给出可理解的提示。"""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = tmp_path / "blank.pdf"
    with pdf.open("wb") as f:
        writer.write(f)

    kb = KnowledgeBase(tmp_path / "kb")
    with pytest.raises(ValueError, match="没有切出任何文字"):
        kb.add_document("blank.pdf", pdf.read_bytes())
    # 失败之后不能留下半成品目录
    assert not (kb.root / "blank").exists()


def test_docx_headings_are_recognised(tmp_path) -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("报告标题", level=1)
    d.add_paragraph("正文内容。" * 20)
    d.add_heading("小节甲", level=2)
    d.add_paragraph("另一段内容。" * 20)
    path = tmp_path / "t.docx"
    d.save(str(path))

    kb = KnowledgeBase(tmp_path / "kb")
    info = kb.add_document("t.docx", path.read_bytes())
    assert info.chunk_count == 2
    # 第一块落在 H1 与 H2 之间（没有小节名），第二块才属于「小节甲」——
    # 这是刻意的：引言不归属任何小节，否则引用会指向一个它并不属于的章节。
    titles = [c.section_title for c in kb.load_chunks()]
    assert titles == ["", "小节甲"]


def test_raw_chunk_returns_text_and_label(tmp_path) -> None:
    """引用深链的落地接口。返回 None 就是 404，所以它必须真的能取到东西。"""
    kb = KnowledgeBase(tmp_path / "kb")
    info = kb.add_document("a.md", ("# 标题\n\n## 小节\n\n" + "正文" * 40).encode())
    cid = kb.load_chunks()[0].chunk_id

    item = kb.raw_chunk(info.doc_id, cid)
    assert item is not None
    assert item["section"] == "小节"
    assert item["label"] == "a · 小节"
    assert item["text"]

    assert kb.raw_chunk(info.doc_id, "不存在的块") is None
    assert kb.raw_chunk("不存在的文档", cid) is None


def test_broken_manifest_does_not_break_the_kb(tmp_path) -> None:
    """清单文件坏了，顶多列表显示不出来，不能让整个知识库不可用。"""
    kb = KnowledgeBase(tmp_path / "kb")
    kb.add_document("a.md", ("# t\n\n" + "正文" * 40).encode())
    kb.manifest_path.write_text("{ 这不是 JSON", encoding="utf-8")
    assert kb.list_documents() == []


# ------------------------------------------------------------------
# 切分公共设施（两个切分器共享，行为必须稳定）
# ------------------------------------------------------------------
def test_norm_text_collapses_whitespace() -> None:
    assert norm_text("  a\n\tb  ") == "a b"
    assert norm_text("\u3000全角　空格\u00a0") == "全角 空格"


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", chapter_id="ch", chapter_no="第 1 章",
        chapter_title="T", part="", section_title="", text=text,
    )


def test_merge_short_absorbs_tiny_chunk() -> None:
    out = merge_short([_chunk(1, "甲" * 100), _chunk(2, "短短短")])
    assert len(out) == 1
    assert "短短短" in out[0].text


def test_merge_short_renumbers_chunk_ids() -> None:
    out = merge_short([_chunk(1, "甲" * 100), _chunk(2, "乙" * 100)])
    assert [c.chunk_id for c in out] == ["ch::c001", "ch::c002"]
