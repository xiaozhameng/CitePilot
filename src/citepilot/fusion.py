"""多路召回的融合：去重 + RRF。

这一层单独拆出来、并且做成纯函数，有两个原因：
1) 它是**算法**而不是工程胶水，纯函数才可被单测钉死（tests/test_fusion.py）。
2) 一个常见的手写实现题就是「输入多组 docId 与排名，输出 RRF 分数排序 Top-K，
   支持按来源权重和去重」—— 这段代码就是那道题的可运行答案。

为什么用 RRF 而不是「分数归一化后加权求和」？
因为不同来源的分数**尺度不可比**：BM25 的分数是无上界的实数（可能 12.7），
向量相似度是 [-1, 1]（可能 0.83）。要加权求和就必须先归一化，
而归一化的分母依赖本次结果集，等于每问一次问题就要重新调一次阈值。
RRF 只看**排名**不看分数，天然免疫尺度问题，而且参数从 5 个降到 1 个（只有 k）。
代价：丢掉了分数的绝对信息（第 1 名和第 2 名的差距被抹平）。
"""

from __future__ import annotations

from collections import defaultdict

# RRF 的平滑常数。60 是原论文的经验值：它让前几名的差距不至于过大，
# 也让"在某一路上排第 1 但另一路完全没召回"的文档不会碾压"两路都排前 10"的文档。
DEFAULT_K = 60


def rrf_fuse(
    ranked_lists: dict[str, list[str]],
    weights: dict[str, float] | None = None,
    k: int = DEFAULT_K,
    top_k: int | None = None,
) -> list[tuple[str, float, dict[str, int]]]:
    """倒数排名融合。

    ranked_lists: {"vec": [chunk_id, ...], "bm25": [...]} 每路已按相关性降序
    weights:      {"vec": 1.0, "bm25": 0.8} 某路权重，缺省 1.0
    k:            平滑常数
    top_k:        截断长度

    返回 [(chunk_id, 融合分, {来源: 名次})]，按融合分降序。
    第三个元素是"归因"，对照实验要靠它解释"这一分是谁贡献的"。
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for source, ids in ranked_lists.items():
        w = weights.get(source, 1.0)
        if w == 0:
            continue
        seen: set[str] = set()
        deduped: list[str] = []
        for doc_id in ids:
            # 去重：同一路里同一个文档出现两次，只算第一次。
            # 注意连排名也要一起跳过 —— 否则重复项虽然不翻倍计分，
            # 却会"占位"，把后面所有文档的名次整体往后推一位，
            # 让它们的分无端变低（测试 test_duplicate_within_one_list_is_counted_once 钉的就是这个）。
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(doc_id)

        for idx, doc_id in enumerate(deduped):
            rank = idx + 1  # RRF 的名次从 1 开始
            scores[doc_id] += w / (k + rank)
            ranks[doc_id][source] = rank

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k:
        fused = fused[:top_k]
    return [(doc_id, score, ranks[doc_id]) for doc_id, score in fused]


def dedupe_by_text(items: list[tuple[str, str]], keep_first: bool = True) -> list[str]:
    """按正文去重：语料里常有完全重复的段落（比如图注与正文重复），
    它们会占掉 Top-K 的名额，让答案里出现两条一模一样的出处。

    输入 [(chunk_id, text)]，返回去重后的 chunk_id 列表（保持原顺序）。
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id, text in items:
        key = "".join(text.split())[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk_id)
    return out
