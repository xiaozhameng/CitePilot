"""评测指标与检索的测试。

指标本身写错的概率比想象中高（NDCG 的折损分子分母搞反、MRR 忘了除以样本数），
而指标错了会导致整个"优化"方向都错。所以用手工可验算的小例子钉住。

另外这里测了 BM25 与意图规则 —— 两者都是"零依赖的确定性组件"，
正因为确定，才值得用测试固定住行为。
"""

import pytest

from citepilot.bm25 import BM25, tokenize
from citepilot.config import SETTINGS
from citepilot.evalkit import (
    _chapter_numbers,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from citepilot.intent import _rule_based


# ---------------- 指标 ----------------
def test_chapter_numbers_parsing():
    assert _chapter_numbers("见第 1 章。") == [1]
    assert _chapter_numbers("见第 0、2 章。") == [0, 2]
    assert _chapter_numbers("见第 3 章和第 5 章。") == [3, 5]
    assert _chapter_numbers("没有出处标注") == []


def test_recall_at_k_hit_any_gold_chapter():
    hits = [["ch-01", "ch-09"], ["ch-07", "ch-08"]]
    golds = [["ch-01"], ["ch-03"]]
    assert recall_at_k(hits, golds, 1) == 0.5
    assert recall_at_k(hits, golds, 2) == 0.5  # 第二条首尾两个都不是 ch-03


def test_recall_at_k_partial_credit_for_k3():
    hits = [["a", "b", "ch-01"]]
    golds = [["ch-01"]]
    assert recall_at_k(hits, golds, 1) == 0.0
    assert recall_at_k(hits, golds, 3) == 1.0


def test_mrr_matches_hand_calculation():
    # 第一条第 1 位命中 → 1.0；第二条第 2 位命中 → 0.5 → 平均 0.75
    hits = [["ch-01", "x"], ["y", "ch-02"]]
    golds = [["ch-01"], ["ch-02"]]
    assert mrr(hits, golds) == pytest.approx(0.75)


def test_mrr_is_zero_when_nothing_hits():
    assert mrr([["a"]], [["b"]]) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    hits = [["ch-01", "ch-02"]]
    golds = [["ch-01", "ch-02"]]
    assert ndcg_at_k(hits, golds, 2) == pytest.approx(1.0)


def test_ndcg_penalizes_late_hits():
    perfect = ndcg_at_k([["ch-01", "x"]], [["ch-01"]], 2)
    late = ndcg_at_k([["x", "ch-01"]], [["ch-01"]], 2)
    assert perfect > late > 0


# ---------------- BM25 ----------------
def test_tokenize_drops_stopwords_and_punct():
    toks = tokenize("请你说明一下，什么是「备付金」？")
    assert "备付金" in toks
    assert "什么" not in toks
    assert "，" not in toks


def test_bm25_ranks_relevant_document_first():
    docs = {
        "d1": "备付金是用户已付、商户未拿的钱，2019 年起 100% 集中缴存央行。",
        "d2": "对账要核三本账：系统账、渠道账、银行账。",
        "d3": "担保交易期间资金处于冻结状态。",
    }
    bm = BM25()
    for k, v in docs.items():
        bm.add(k, v)
    bm.build()
    top = bm.search("备付金缴存央行", top_k=1)
    assert top and top[0][0] == "d1"


def test_bm25_returns_nothing_for_unrelated_query():
    bm = BM25()
    bm.add("d1", "备付金集中缴存")
    bm.build()
    assert bm.search("量子纠缠退相干", top_k=3) == []


def test_bm25_boost_text_makes_chapter_name_searchable():
    """索引文本里带上章节标签，'第 8 章'这类查询才能字面命中。"""
    bm = BM25()
    bm.add("d1", "正文与网联无关", boost="第 8 章 银联 · 数字人民币 · 网联")
    bm.add("d2", "另一段完全无关的内容", boost="第 1 章 支付的本质")
    bm.build()
    assert bm.search("网联", top_k=1)[0][0] == "d1"


# ---------------- 意图规则层 ----------------
def test_rule_layer_handles_greeting_without_model():
    r = _rule_based("你好")
    assert r and r.route == "direct" and r.layer == "L0_rule"


def test_rule_layer_refuses_fund_operations():
    r = _rule_based("帮我转账给张三 100 块")
    assert r and r.route == "refuse"


def test_rule_layer_passes_real_questions_to_next_layer():
    assert _rule_based("备付金是什么？") is None


def test_rule_layer_catches_empty_input():
    assert _rule_based("") is not None
    assert _rule_based("？") is not None


# ---------------- 计价 ----------------
def test_cost_calculation_is_exact():
    s = SETTINGS
    cost = s.cost_cny(1_000_000, 1_000_000)
    assert cost == pytest.approx(s.price_in_per_m + s.price_out_per_m)
    assert s.cost_cny(0, 0) == 0.0
