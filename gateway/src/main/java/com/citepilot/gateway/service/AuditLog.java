package com.citepilot.gateway.service;

import com.citepilot.gateway.config.GatewayProperties;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * 审计日志。每轮问答落一条 JSONL。
 *
 * <p>为什么网关要单独做一份审计，而不是直接读内核的 trace 库？
 *
 * <p>因为两者回答的是不同问题：
 * <ul>
 *   <li><b>内核的 trace</b> 回答"这次问答内部发生了什么" ——
 *       意图分流耗时、检索命中几条、引用是否校验通过。它关心的是算法。</li>
 *   <li><b>网关的审计</b> 回答"谁在什么时候用了多少" ——
 *       客户端是谁、走的哪条策略、有没有被限流、一共消耗多少 token。
 *       它关心的是运营与配额。</li>
 * </ul>
 *
 * <p>混在一张表里会有两个后果：内核要知道"用户是谁"（越界了），
 * 且审计写失败会影响问答主链路（不该）。所以两边各记各的，
 * 靠 traceId 关联。这也是"审计与业务解耦"的常规做法。
 *
 * <p>格式选 JSONL 不选数据库，是刻意的：一行一条，
 * 可以直接 {@code grep}、{@code tail -f}、也能一把导入任何分析工具。
 * 演示时打开终端 tail 一下，日志自己往外滚，比点开数据库客户端有说服力。
 */
@Service
public class AuditLog {

    private static final Logger log = LoggerFactory.getLogger(AuditLog.class);

    private final GatewayProperties.Audit cfg;
    private final ObjectMapper mapper;
    private final Path path;
    private final Deque<ObjectNode> memory = new ArrayDeque<>();
    private BufferedWriter writer;

    public AuditLog(GatewayProperties props, ObjectMapper mapper) {
        this.cfg = props.getAudit();
        this.mapper = mapper;
        this.path = Path.of(cfg.getPath()).toAbsolutePath();
        if (cfg.isEnabled()) {
            try {
                Files.createDirectories(path.getParent());
                this.writer = Files.newBufferedWriter(path,
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException e) {
                // 审计写不了不该拖垮服务。降级为只留内存，并明确告警。
                log.error("审计文件无法打开（{}），本次进程只保留内存审计", path, e);
            }
        }
    }

    /**
     * 记一条。
     *
     * <p>全方法体包在 try 里且不往外抛 —— 审计是旁路，
     * 旁路失败绝不能影响主链路。这是审计这类组件的第一原则。
     */
    public synchronized void record(ObjectNode entry) {
        entry.put("gatewayTs", System.currentTimeMillis());
        // 内存环形缓冲：演示页要展示"最近 N 条"，不必每次都去读文件
        memory.addFirst(entry.deepCopy());
        while (memory.size() > cfg.getMemoryBuffer()) {
            memory.removeLast();
        }
        if (writer != null) {
            try {
                writer.write(mapper.writeValueAsString(entry));
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                log.warn("审计写入失败：{}", e.toString());
            }
        }
    }

    /** 从内核的完整答案里抽字段，拼成一条审计记录。 */
    public ObjectNode fromAnswer(String clientKey, String sessionId, String question,
                                 String strategy, boolean useRerank,
                                 boolean streamed, boolean revoked, JsonNode answer) {
        ObjectNode e = mapper.createObjectNode();
        e.put("clientKey", clientKey == null ? "-" : clientKey);
        e.put("sessionId", sessionId == null ? "-" : sessionId);
        e.put("question", question);
        e.put("strategy", strategy);
        e.put("useRerank", useRerank);
        e.put("streamed", streamed);
        e.put("revoked", revoked);
        if (answer == null) {
            e.put("ok", false);
            e.put("error", "no_answer");
            return e;
        }
        e.put("ok", true);
        e.put("route", answer.path("route").asText(""));
        e.put("grounded", answer.path("grounded").asBoolean(false));
        e.put("degraded", answer.path("degraded").asText(""));
        e.put("rewrites", answer.path("rewrites").asInt(0));
        e.put("intent", answer.path("intent").asText(""));
        e.put("intentConfidence", answer.path("intent_confidence").asDouble(0));
        e.put("citationCount", answer.path("citations").size());

        JsonNode m = answer.path("metrics");
        e.put("latencyMs", m.path("latency_ms").asInt(0));
        e.put("totalMs", m.path("total_ms").asInt(0));
        e.put("promptTokens", m.path("prompt_tokens").asInt(0));
        e.put("completionTokens", m.path("completion_tokens").asInt(0));
        e.put("answerChars", answer.path("text").asText("").length());
        return e;
    }

    /** 最近 N 条（内存里的），新的在前。 */
    public List<JsonNode> recent(int limit) {
        synchronized (this) {
            List<JsonNode> out = new ArrayList<>();
            int i = 0;
            for (ObjectNode n : memory) {
                if (i++ >= limit) break;
                out.add(n);
            }
            return out;
        }
    }

    public String pathString() {
        return path.toString();
    }

    public boolean fileEnabled() {
        return writer != null;
    }

    /**
     * 关流。
     *
     * <p>必须有这一步：BufferedWriter 里可能还有没落盘的数据，
     * 进程退出时不 close 就会丢最后几条 —— 而恰恰是"出问题前的那几条"
     * 最有价值。
     */
    @PreDestroy
    public synchronized void close() {
        if (writer != null) {
            try {
                writer.flush();
                writer.close();
            } catch (IOException e) {
                log.warn("审计文件关闭失败：{}", e.toString());
            }
            writer = null;
        }
    }
}
