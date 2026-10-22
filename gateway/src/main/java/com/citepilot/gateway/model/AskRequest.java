package com.citepilot.gateway.model;

/**
 * 问答请求。
 *
 * @param question  用户问题（必填）
 * @param strategy  召回策略：hybrid / bm25_only / vec_only。默认 hybrid。
 *                  把它暴露出来是为了让演示页能现场切换做对照实验 ——
 *                  点两下就能把"BM25 vs 向量 vs 混合"的差异跑出来对比。
 * @param useRerank 是否开启精排。默认 true。
 * @param sessionId 会话 id。不传则视为单轮问答（网关不记录上下文）。
 */
public record AskRequest(
        String question,
        String strategy,
        Boolean useRerank,
        String sessionId
) {
    public String strategyOrDefault() {
        return (strategy == null || strategy.isBlank()) ? "hybrid" : strategy;
    }

    public boolean useRerankOrDefault() {
        return useRerank == null || useRerank;
    }
}
