"""纯 Python BM25 —— 零依赖的稀疏检索兜底路。

为什么要自己写一个而不是装 rank_bm25？
两个原因：一是 BM25 只有 30 行，自己写可控（能加同义词、能改分词）；
二是它是整个系统的**降级底座** —— 向量服务挂了、embedding 服务没起来、
甚至完全断网，这条路必须在。"降级链"在这里不是概念，是这段代码。

中文必须分词，否则按字切会把「支付机构」拆成四个不相干的字。
这里用 jieba 的搜索引擎模式，同时把术语表里的词加进自定义词典。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import jieba

# 关掉 jieba 的初始化日志，否则第一次调用会往 stdout 打印一堆 Loading model cost...
jieba.setLogLevel(60)

# 这些词在每个文档里都出现，不具备区分度，但会拉高分数
STOPWORDS = set(
    "的 了 和 与 或 是 在 有 不 也 都 就 而 及 等 这 那 你 我 他 它 们 一 个 中 上 下 为 以 于 对 从 到 被 把 让 "
    "吗 呢 吧 啊 呀 什么 怎么 如何 为什么 哪些 哪个 请 说明 解释 介绍".split()
)

# 领域词典：这些词必须整体成词，绝不能被切开。
#
# 为什么必须显式加？——因为 jieba 的通用词典里没有"备付金"，
# 它会把「备付金」切成 ['备付', '金']。后果不是"差一点"，是灾难性的：
#   1) 用户问"备付金是什么"，文档里写的是"备付金"，两边都切碎还碰巧能对上；
#   2) 但一旦用户写"备付"，就会命中一堆含"备付"的无关段落，
#      而真正的定义块因为被切成了"备付"+"金"反而排到后面。
# 这是"字面检索在专业领域失灵"的最典型原因，也是通用分词器
# 在垂直场景必须补词典的理由。
DOMAIN_TERMS = [
    "备付金", "断直连", "二清", "收单机构", "发卡行", "收单行", "卡组织", "清算", "结算",
    "轧差", "划拨", "头寸", "日切", "长款", "短款", "对账", "分账", "掉单", "关单",
    "异步通知", "回调", "担保交易", "支付中台", "支付机构", "清算机构", "网联", "银联",
    "数字人民币", "聚合支付", "扫码支付", "商户号", "单边账", "幂等", "重复支付",
    "资金账单", "交易单", "支付单", "退款", "退券", "分账回退", "T+1", "D+0",
    "聚合码", "收银台", "风控", "资损", "三流合一", "信息流", "资金流", "货物流",
]


def register_terms(terms: list[str] | None = None) -> None:
    """把领域术语加进 jieba 词典。idempotent，重复调用无副作用。

    语料自带的术语表也可以从这里注入 —— 让分词器知道这个领域里
    "什么是一个词"，是垂直检索的第一道调优。
    """
    for t in (terms or DOMAIN_TERMS):
        t = t.strip()
        if t:
            jieba.add_word(t, freq=10_000)


register_terms()

_PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uffef!-/:-@\[-`{-~]+")


def tokenize(text: str) -> list[str]:
    """中文分词 + 英文数字保底。返回过滤掉停用词与标点的 token 列表。"""
    text = text.lower()
    tokens: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = _PUNCT.sub("", tok).strip()
        if not tok or tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


class BM25:
    """标准 Okapi BM25。

    公式：score(D,Q) = Σ IDF(q) · f(q,D)(k1+1) / ( f(q,D) + k1(1-b+b·|D|/avgdl) )
    直觉：词在本文档出现越多越相关（f），但出现太多次收益递减（k1），
         长文档要打个折，免得"什么都长"的文档占便宜（b）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []                      # 文档 id（即 chunk_id）
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.tf: list[Counter] = []                    # 每篇文档的词频
        self.df: Counter = Counter()                   # 每个词出现在多少篇文档里
        self._built = False

    def add(self, doc_id: str, text: str, boost: str = "") -> None:
        """boost 是给检索用的加权文本（比如「章标题 + 正文」），
        让「第 3 章」这样的章名也能被命中。"""
        toks = tokenize(f"{boost} {text}" if boost else text)
        self.docs.append(doc_id)
        self.doc_tokens.append(toks)
        self.doc_len.append(len(toks))
        self.tf.append(Counter(toks))

    def build(self) -> None:
        self.df = Counter()
        for counter in self.tf:
            self.df.update(counter.keys())
        n = len(self.docs)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self._built = True

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        # +1 的平滑项保证高频词 idf 也为正，不会出现"命中反而扣分"
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, score)]，按分数降序。"""
        if not self._built:
            self.build()
        q_terms = set(tokenize(query))
        scores: list[tuple[str, float]] = []
        for i, counter in enumerate(self.tf):
            s = 0.0
            dl = self.doc_len[i] or 1
            for term in q_terms:
                f = counter.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((self.docs[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    @classmethod
    def from_chunks(cls, chunks: list) -> BM25:
        bm = cls()
        for c in chunks:
            # 索引文本 = 出处标签（章·节）+ 正文。让章节能被字面命中。
            bm.add(c.chunk_id, c.text, boost=c.label)
        bm.build()
        return bm
