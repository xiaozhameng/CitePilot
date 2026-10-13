package com.citepilot.gateway.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

/**
 * 网关配置。全部可在 application.yml 里改，也可被环境变量覆盖
 * （Spring Boot 的宽松绑定：GATEWAY_KERNEL_BASE_URL 覆盖 gateway.kernel.base-url）。
 *
 * <p>把 upstream 的 token 做成配置项而不是硬编码，是为了做到
 * "换内核地址不改代码" —— 内核从本机挪到内网另一台机器时，
 * 只改一行 yml。
 */
@ConfigurationProperties(prefix = "gateway")
public class GatewayProperties {

    private final Kernel kernel = new Kernel();
    private final RateLimit rateLimit = new RateLimit();
    private final Session session = new Session();
    private final Audit audit = new Audit();

    public Kernel getKernel() { return kernel; }
    public RateLimit getRateLimit() { return rateLimit; }
    public Session getSession() { return session; }
    public Audit getAudit() { return audit; }

    /** 上游 Python 内核。 */
    public static class Kernel {
        /** 内核地址。默认只在本机，说明网关和内核当前是同居部署。 */
        private String baseUrl = "http://127.0.0.1:8077";
        /** 与内核约定的共享 token，对应内核的 CITEPILOT_INTERNAL_TOKEN。 */
        private String token = "dev-local-token-change-me";
        /** 连接超时。连都连不上就没必要等。 */
        private Duration connectTimeout = Duration.ofSeconds(3);
        /** 单次问答的读超时。内核 P95 约 20 秒，留一倍余量。 */
        private Duration readTimeout = Duration.ofSeconds(60);
        /** 流式响应的空闲超时：超过这个时长没收到任何事件才判定掉线。 */
        private Duration streamIdleTimeout = Duration.ofSeconds(45);
        /** WebClient 连接池上限。内核是单进程，给多了只会把上游压垮。 */
        private int maxConnections = 64;

        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getToken() { return token; }
        public void setToken(String token) { this.token = token; }
        public Duration getConnectTimeout() { return connectTimeout; }
        public void setConnectTimeout(Duration v) { this.connectTimeout = v; }
        public Duration getReadTimeout() { return readTimeout; }
        public void setReadTimeout(Duration v) { this.readTimeout = v; }
        public Duration getStreamIdleTimeout() { return streamIdleTimeout; }
        public void setStreamIdleTimeout(Duration v) { this.streamIdleTimeout = v; }
        public int getMaxConnections() { return maxConnections; }
        public void setMaxConnections(int v) { this.maxConnections = v; }
    }

    /**
     * 限流参数。
     *
     * <p>为什么要限流？因为内核一次问答要 7~12 秒，且它是有状态的单进程。
     * 不限流的话，几十个并发打进来，每个请求都要排队，
     * 结果是所有人的延迟一起涨 —— 这是"雪崩"的经典形态。
     * 宁可明确拒绝一部分（429），也不让所有人都变慢。
     */
    public static class RateLimit {
        private boolean enabled = true;
        /** 令牌桶容量：允许的突发量。 */
        private int burst = 6;
        /** 每秒补充的令牌数，即稳态 QPS 上限。 */
        private double refillPerSecond = 0.5;
        /** 同时进行的流式请求上限（比 QPS 更硬的一道闸）。 */
        private int maxConcurrentStreams = 8;

        public boolean isEnabled() { return enabled; }
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        public int getBurst() { return burst; }
        public void setBurst(int burst) { this.burst = burst; }
        public double getRefillPerSecond() { return refillPerSecond; }
        public void setRefillPerSecond(double v) { this.refillPerSecond = v; }
        public int getMaxConcurrentStreams() { return maxConcurrentStreams; }
        public void setMaxConcurrentStreams(int v) { this.maxConcurrentStreams = v; }
    }

    /** 会话：让"多轮追问"有地方落。内核自身无状态，会话归网关管。 */
    public static class Session {
        /** 单个会话保留的最大轮次。 */
        private int maxTurns = 20;
        /** 会话闲置多久后回收。 */
        private Duration idleTimeout = Duration.ofMinutes(30);
        /** 同时保留的会话数上限（内存版实现必须给自己设个天花板）。 */
        private int maxSessions = 500;

        public int getMaxTurns() { return maxTurns; }
        public void setMaxTurns(int maxTurns) { this.maxTurns = maxTurns; }
        public Duration getIdleTimeout() { return idleTimeout; }
        public void setIdleTimeout(Duration idleTimeout) { this.idleTimeout = idleTimeout; }
        public int getMaxSessions() { return maxSessions; }
        public void setMaxSessions(int v) { this.maxSessions = v; }
    }

    /** 审计。落 JSONL 文件，一个文件一行，方便直接 grep 与导入。 */
    public static class Audit {
        private boolean enabled = true;
        private String path = "var/gateway-audit.jsonl";
        /** 内存里保留多少条供演示页展示。 */
        private int memoryBuffer = 200;

        public boolean isEnabled() { return enabled; }
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        public String getPath() { return path; }
        public void setPath(String path) { this.path = path; }
        public int getMemoryBuffer() { return memoryBuffer; }
        public void setMemoryBuffer(int v) { this.memoryBuffer = v; }
    }
}
