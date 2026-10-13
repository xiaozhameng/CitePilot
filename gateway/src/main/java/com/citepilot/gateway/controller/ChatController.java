package com.citepilot.gateway.controller;

import com.citepilot.gateway.model.AskRequest;
import com.citepilot.gateway.service.AuditLog;
import com.citepilot.gateway.service.KernelClient;
import com.citepilot.gateway.service.RateLimiter;
import com.citepilot.gateway.service.SessionStore;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 对外的问答入口。两条路：
 * <ul>
 *   <li>POST /api/ask        等等等 → 一次性拿完整 JSON</li>
 *   <li>POST /api/ask/stream 边算边推 → SSE</li>
 * </ul>
 *
 * <p><b>为什么两条都要留？</b>
 * 流式是给"人"用的（补一个首字节延迟的体验），
 * 非流式是给"机器"用的（脚本、批量评测、别的系统集成）。
 * 只留流式，脚本要自己解析 SSE，很别扭；
 * 只留非流式，交互式界面要盯 10 秒空屏。
 * 两条路共用同一个 KernelClient，成本很低。
 */
@RestController
@RequestMapping("/api")
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final KernelClient kernel;
    private final RateLimiter limiter;
    private final SessionStore sessions;
    private final AuditLog audit;
    private final ObjectMapper mapper;

    public ChatController(KernelClient kernel, RateLimiter limiter,
                          SessionStore sessions, AuditLog audit, ObjectMapper mapper) {
        this.kernel = kernel;
        this.limiter = limiter;
        this.sessions = sessions;
        this.audit = audit;
        this.mapper = mapper;
    }

    /**
     * 客户端标识：优先用显式传入的会话，其次用远端地址。
     *
     * <p>限流按这个维度分桶，而不是全局一个桶 ——
     * 全局桶会让一个刷接口的人把所有人挤下线。
     */
    private String clientKey(ServerWebExchange ex, AskRequest req) {
        if (req != null && req.sessionId() != null && !req.sessionId().isBlank()) {
            return "s:" + req.sessionId();
        }
        var remote = ex.getRequest().getRemoteAddress();
        if (remote != null && remote.getAddress() != null) {
            return "ip:" + remote.getAddress().getHostAddress();
        }
        return "anon";
    }

    // ------------------------------------------------------------------
    // 非流式
    // ------------------------------------------------------------------
    @PostMapping("/ask")
    public Mono<Object> ask(@RequestBody AskRequest req, ServerWebExchange ex) {
        if (req == null || req.question() == null || req.question().isBlank()) {
            return Mono.just(Map.of("error", "question 不能为空"));
        }
        String key = clientKey(ex, req);
        if (!limiter.tryAcquire(key)) {
            ex.getResponse().setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
            return Mono.just(Map.of("error", "请求过于频繁，请稍后再试"));
        }

        var session = sessions.getOrCreate(req.sessionId());

        return kernel.ask(req)
                .map(node -> {
                    ObjectNode entry = audit.fromAnswer(key, session.id(), req.question(),
                            req.strategyOrDefault(), req.useRerankOrDefault(), false, false, node);
                    audit.record(entry);
                    sessions.record(session.id(),
                            new SessionStore.Turn(req.question(),
                                    node.path("text").asText("").substring(
                                            0, Math.min(80, node.path("text").asText("").length())),
                                    node.path("route").asText(""),
                                    node.path("grounded").asBoolean(false),
                                    node.path("metrics").path("latency_ms").asInt(0),
                                    System.currentTimeMillis()));
                    return (Object) Map.of(
                            "sessionId", session.id(),
                            "answer", node);
                })
                .onErrorResume(e -> {
                    log.error("非流式问答失败", e);
                    return Mono.just(Map.of("error", "上游处理失败：" + e.getMessage()));
                });
    }

    // ------------------------------------------------------------------
    // 流式（SSE）
    // ------------------------------------------------------------------
    @PostMapping(value = "/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> askStream(@RequestBody AskRequest req, ServerWebExchange ex) {
        if (req == null || req.question() == null || req.question().isBlank()) {
            return Flux.just(sse("error", "{\"message\":\"question 不能为空\"}"));
        }

        String key = clientKey(ex, req);
        ServerHttpResponse response = ex.getResponse();

        // 闸一：速率
        if (!limiter.tryAcquire(key)) {
            response.setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
            return Flux.just(sse("error", "{\"message\":\"请求过于频繁，请稍后再试\"}"));
        }
        // 闸二：并发。占不到就立刻拒绝，绝不排队 ——
        // 排队会让所有请求一起变慢，比直接拒绝更糟。
        if (!limiter.tryAcquireStreamSlot()) {
            response.setStatusCode(HttpStatus.SERVICE_UNAVAILABLE);
            return Flux.just(sse("error",
                    "{\"message\":\"当前并发已满（内核是单实例，为保护延迟而限流），请稍后重试\"}"));
        }

        var session = sessions.getOrCreate(req.sessionId());

        // 把网关分配好的 sessionId 回传给内核。
        // 内核自己不知道会话这回事 —— 会话是网关的概念。
        // 但如果不下传，内核的 trace 里 session 永远是空的，
        // 排查"某个会话为什么答得差"时就串不起来。多传一个字段就能打通这条线。
        AskRequest reqWithSession = new AskRequest(
                req.question(), req.strategy(), req.useRerank(), session.id());

        // 审计状态在流过程中逐步补齐：revoked 要等 revoke 事件，answer 要等最后一个事件。
        // 用 Atomic 是因为这几个字段会跨事件写、在流结束时读。
        AtomicBoolean revoked = new AtomicBoolean(false);
        AtomicBoolean answered = new AtomicBoolean(false);

        Flux<ServerSentEvent<String>> upstream = kernel.stream(reqWithSession)
                .doOnNext(ev -> {
                    String name = ev.event() == null ? "message" : ev.event();
                    if ("revoke".equals(name)) {
                        revoked.set(true);
                    }
                    if ("answer".equals(name)) {
                        answered.set(true);
                        try {
                            JsonNode payload = mapper.readTree(ev.data());
                            JsonNode answer = payload.path("answer");
                            audit.record(audit.fromAnswer(key, session.id(), req.question(),
                                    req.strategyOrDefault(), req.useRerankOrDefault(),
                                    true, revoked.get(), answer));
                            String text = answer.path("text").asText("");
                            sessions.record(session.id(), new SessionStore.Turn(
                                    req.question(),
                                    text.substring(0, Math.min(80, text.length())),
                                    answer.path("route").asText(""),
                                    answer.path("grounded").asBoolean(false),
                                    answer.path("metrics").path("latency_ms").asInt(0),
                                    System.currentTimeMillis()));
                        } catch (Exception e) {
                            log.warn("审计解析失败（不影响主链路）：{}", e.toString());
                        }
                    }
                })
                .doFinally(signal -> {
                    // 客户端中途断开时上游不会发 answer，这里补一条，
                    // 否则"用户看到一半关掉了"这件事在审计里完全不可见。
                    if (!answered.get()) {
                        audit.record(audit.fromAnswer(key, session.id(), req.question(),
                                req.strategyOrDefault(), req.useRerankOrDefault(),
                                true, revoked.get(), null)
                                .put("aborted", true)
                                .put("signal", signal.toString()));
                    }
                    // 并发位必须无条件释放。漏一次，系统就永久少一个并发位，
                    // 而且这种泄漏只会表现为"越来越慢"，极难排查。
                    limiter.releaseStreamSlot();
                });

        // 在最前面插一个网关自己的事件，把分配到的 sessionId 告诉前端。
        // 内核也会发 open（带 session_id），但那个值可能是空的 ——
        // 会话是网关的概念，应该由网关宣告。
        Flux<ServerSentEvent<String>> header = Flux.just(sse("session",
                "{\"sessionId\":\"" + session.id() + "\",\"slotsLeft\":"
                        + limiter.availableStreamSlots() + "}"));

        return Flux.concat(header, upstream);
    }

    private static ServerSentEvent<String> sse(String event, String data) {
        return ServerSentEvent.<String>builder().event(event).data(data).build();
    }
}
