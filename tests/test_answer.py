"""引用校验的单元测试 —— 这是"答案必带出处"的最后一道防线。

需求是"给出答案时提供出处"，但真正难的是：**怎么知道模型标的出处是真的？**
模型完全可能写出 `[7]` 而本次只召回了 5 条资料。这不是格式问题，是幻觉问题。
所以校验逻辑必须有测试兜住，否则"带出处"只是个装饰。
"""

from citepilot.answer import (
    AnswerGenerator,
    build_context,
    parse_markers,
    validate_markers,
    _extractive_answer,
)
from citepilot.schema import Chunk, Hit


def _hit(i: int, text: str = "正文内容") -> Hit:
    c = Chunk(
        chunk_id=f"ch-01::c{i:03d}", chapter_id="ch-01", chapter_no="第 1 章",
        chapter_title="支付的本质与全景", part="第一篇", section_title="1.1 三流合一",
        text=text,
    )
    return Hit(chunk=c, score=1.0 / i)


def test_parse_markers_keeps_first_seen_order_and_dedupes():
    assert parse_markers("结论一[2]，结论二[1]，又引用[2]和[4]") == [2, 1, 4]


def test_parse_markers_handles_adjacent_markers():
    assert parse_markers("备付金已于 2019 年全额缴存央行[1][3]") == [1, 3]


def test_validate_rejects_out_of_range_marker():
    """越界引用 = 引用幻觉，必须被拦下。"""
    ok, problem = validate_markers("这句话引用了不存在的来源[7]。", n_hits=5)
    assert not ok
    assert "7" in problem


def test_validate_rejects_answer_without_any_citation():
    """写了事实却不标出处，同样不合格 —— 否则"带出处"这个需求等于没做。"""
    ok, problem = validate_markers("备付金是集中缴存的。", n_hits=5)
    assert not ok
    assert "角标" in problem


def test_validate_accepts_valid_citations():
    ok, problem = validate_markers("备付金全额缴存央行[1]，详见第 3 章[2]。", n_hits=3)
    assert ok and problem == ""


def test_build_context_numbers_sources_from_one():
    ctx = build_context([_hit(1, "甲"), _hit(2, "乙")])
    assert ctx.startswith("[1] 出处：第 1 章")
    assert "[2] 出处：第 1 章" in ctx


def test_extractive_fallback_never_fabricates():
    """降级路径的引用必须 100% 是原文，因为答案就是原文本身。"""
    text, cits = _extractive_answer("随便问点什么", [_hit(1, "第一段原文"), _hit(2, "第二段原文")])
    assert "第一段原文" in text
    assert [c.marker for c in cits] == [1, 2]
    assert all(c.quote for c in cits)


def test_extractive_fallback_with_no_hits():
    text, cits = _extractive_answer("问", [])
    assert "没有检索到" in text
    assert cits == []


def test_generator_without_llm_degrades_instead_of_failing():
    """模型不可用时，系统仍然要能给出带出处的答案（摘录模式），而不是报错。"""

    class _NoLLM:
        enabled = False

    gen = AnswerGenerator(llm=_NoLLM())
    res = gen.generate("备付金是什么", [_hit(1, "备付金的定义原文")])
    assert res.answer.degraded == "llm_unavailable_extractive"
    assert res.answer.citations and res.answer.citations[0].marker == 1


def test_generator_with_no_hits_refuses_to_call_model():
    """召回为空时不该浪费一次模型调用去编答案。"""

    class _Boom:
        enabled = True

        def complete(self, *a, **k):  # pragma: no cover - 不应被调用
            raise AssertionError("召回为空时不应调用模型")

    gen = AnswerGenerator(llm=_Boom())
    res = gen.generate("问题", [])
    assert res.answer.route == "refuse"
    assert res.answer.degraded == "no_hits"
