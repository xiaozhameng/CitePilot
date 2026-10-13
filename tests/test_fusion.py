"""RRF 融合的单元测试。

为什么这个模块的测试最值得写？
因为它对应一道常见的手写实现题（"实现多路召回结果融合：输入多组 docId 与排名，
输出 RRF 分数排序 Top-K，支持按来源权重和去重"）。把测试写死，
等于把这道题的边界情况提前想清楚了：

  · 两路都排第一的文档，一定排在只在一路排第一的文档前面吗？
  · 同一路里同一个文档重复出现，会不会把分数刷上去？（本题的坑）
  · 把某一路权重设为 0，它还会不会影响结果？
  · k 越大，排名的区分度是变强还是变弱？
"""

from citepilot.fusion import DEFAULT_K, dedupe_by_text, rrf_fuse


def test_two_way_both_ranked_beats_single_way():
    """两路都出现的文档，应该赢过只在一路出现的文档 —— 这是 RRF 的核心价值。"""
    fused = rrf_fuse({"vec": ["a", "b"], "bm25": ["b", "a"]}, k=60)
    ids = [x[0] for x in fused]
    assert set(ids) == {"a", "b"}
    # a 与 b 在两路里名次互换，融合分应当完全相等
    assert abs(fused[0][1] - fused[1][1]) < 1e-12


def test_single_source_document_lower_than_double_source():
    """只在 BM25 出现的文档，分数必须低于两路都出现的文档。"""
    fused = dict((cid, s) for cid, s, _ in rrf_fuse({"vec": ["a"], "bm25": ["b", "a"]}, k=60))
    assert fused["a"] > fused["b"]


def test_duplicate_within_one_list_is_counted_once():
    """同一路里重复出现只算第一次。

    这不是吹毛求疵：语料切分有重叠时，同一段文本可能产生两个 chunk，
    如果两个 chunk 在同一路里都排前面，重复计数会让它"凭空"拿到双倍分，
    把真正相关的另一条挤下去。
    """
    once = dict((c, s) for c, s, _ in rrf_fuse({"bm25": ["a", "b"]}, k=60))
    twice = dict((c, s) for c, s, _ in rrf_fuse({"bm25": ["a", "a", "b"]}, k=60))
    assert once["a"] == twice["a"]
    assert once["b"] == twice["b"]


def test_source_attribution():
    """融合结果要能说清"这一分是谁贡献的"，对照实验靠它解释归因。"""
    fused = rrf_fuse({"vec": ["a"], "bm25": ["a", "b"]}, k=60)
    ranks = {cid: r for cid, _, r in fused}
    assert ranks["a"] == {"vec": 1, "bm25": 1}
    assert ranks["b"] == {"bm25": 2}


def test_zero_weight_disables_a_source():
    fused = dict((c, s) for c, s, _ in rrf_fuse({"vec": ["a"], "bm25": ["b"]}, weights={"vec": 0}, k=60))
    assert "a" not in fused
    assert "b" in fused


def test_top_k_truncates():
    fused = rrf_fuse({"bm25": ["a", "b", "c", "d"]}, top_k=2, k=60)
    assert len(fused) == 2


def test_larger_k_compresses_the_gap():
    """k 是"名次差距的放大器"：k 越大，第一名和第二名的分数差越小。"""
    small = rrf_fuse({"bm25": ["a", "b"]}, k=1)
    large = rrf_fuse({"bm25": ["a", "b"]}, k=1000)
    gap_small = small[0][1] - small[1][1]
    gap_large = large[0][1] - large[1][1]
    assert gap_large < gap_small
    assert DEFAULT_K == 60


def test_dedupe_by_text_keeps_first():
    items = [("c1", "同样的内容"), ("c2", "另一段"), ("c3", "同样的内容  ")]
    assert dedupe_by_text(items) == ["c1", "c2"]


def test_empty_input():
    assert rrf_fuse({}) == []
