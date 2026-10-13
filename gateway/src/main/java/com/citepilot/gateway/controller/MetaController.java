package com.citepilot.gateway.controller;

import com.citepilot.gateway.config.GatewayProperties;
import com.citepilot.gateway.service.AuditLog;
import com.citepilot.gateway.service.KernelClient;
import com.citepilot.gateway.service.RateLimiter;
import com.citepilot.gateway.service.SessionStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 运维与观测入口。演示页右侧的面板读的就是这几个接口。
 *
 * <p>这里把「网关自己的状态」和「内核的状态」分开返回，
 * 是为了在出问题时能一眼定位是哪一层的锅：
 * 网关的并发位满了 → 是流量问题；
 * 内核 health 里 embed 降级了 → 是外部依赖问题。
 */
@RestController
@RequestMapping("/api")
public class MetaController {

    private final KernelClient kernel;
    private final RateLimiter limiter;
    private final SessionStore sessions;
    private final AuditLog audit;
    private final GatewayProperties props;

    public MetaController(KernelClient kernel, RateLimiter limiter, SessionStore sessions,
                          AuditLog audit, GatewayProperties props) {
        this.kernel = kernel;
        this.limiter = limiter;
        this.sessions = sessions;
        this.audit = audit;
        this.props = props;
    }

    /** 网关 + 内核的组合健康状态。 */
    @GetMapping("/health")
    public Mono<Map<String, Object>> health() {
        Map<String, Object> gateway = new LinkedHashMap<>();
        gateway.put("status", "UP");
        gateway.put("streamSlotsAvailable", limiter.availableStreamSlots());
        gateway.put("streamSlotsMax", props.getRateLimit().getMaxConcurrentStreams());
        gateway.put("sessions", sessions.size());
        gateway.put("auditFile", audit.fileEnabled() ? audit.pathString() : "disabled");
        gateway.put("rateLimitKeys", limiter.trackedKeys());
        gateway.put("kernelBaseUrl", props.getKernel().getBaseUrl());

        return kernel.health()
                .map(k -> {
                    Map<String, Object> out = new LinkedHashMap<>();
                    out.put("gateway", gateway);
                    out.put("kernel", k);
                    out.put("ok", true);
                    return out;
                })
                .onErrorResume(e -> {
                    Map<String, Object> out = new LinkedHashMap<>();
                    gateway.put("status", "UP");
                    out.put("gateway", gateway);
                    out.put("kernel", Map.of("ok", false, "error", e.getMessage()));
                    out.put("ok", false);
                    out.put("checkedAt", Instant.now().toString());
                    return Mono.just(out);
                });
    }

    /** 内核的问答聚合指标（P50 / P95 / 成本 / 降级分布）。 */
    @GetMapping("/stats")
    public Mono<Map<String, Object>> stats() {
        return kernel.stats()
                .map(s -> {
                    Map<String, Object> out = new LinkedHashMap<>();
                    out.put("kernel", s);
                    out.put("gateway", Map.of(
                            "streamSlotsAvailable", limiter.availableStreamSlots(),
                            "sessions", sessions.size()));
                    return out;
                })
                .onErrorResume(e -> Mono.just(Map.of("error", "取内核指标失败：" + e.getMessage())));
    }

    /**
     * 审计流水。
     *
     * @param source memory=只看本次进程内（快）；file=读落盘文件（能看到更久的历史）。
     */
    @GetMapping("/recent")
    public Mono<Map<String, Object>> recent(
            @RequestParam(defaultValue = "30") int limit,
            @RequestParam(defaultValue = "memory") String source) {

        if ("file".equalsIgnoreCase(source)) {
            return kernel.recent(limit)
                    .map(r -> Map.<String, Object>of("source", "kernel-trace", "items", r))
                    .onErrorResume(e -> Mono.just(Map.of("error", e.getMessage())));
        }
        List<?> items = audit.recent(limit);
        return Mono.just(Map.of(
                "source", "gateway-audit(memory)",
                "auditFile", audit.fileEnabled() ? audit.pathString() : "disabled",
                "items", items));
    }

    /** 当前所有活跃会话。 */
    @GetMapping("/sessions")
    public Mono<Map<String, Object>> sessionList() {
        var list = sessions.list().stream().map(sessions::describe).toList();
        return Mono.just(Map.of("count", list.size(), "items", list));
    }
}
