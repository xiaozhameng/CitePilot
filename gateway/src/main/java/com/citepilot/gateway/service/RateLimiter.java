package com.citepilot.gateway.service;

import com.citepilot.gateway.config.GatewayProperties;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 限流。两道不同性质的闸，缺一不可。
 *
 * <p><b>闸一：令牌桶（管速率）。</b>
 * 允许突发，但长期速率受限。实现用的是「懒补充」——
 * 不跑定时任务，每次请求时按时间差补令牌。
 * 这样零后台线程，也不用担心定时任务被 GC 掉或者重启丢状态。
 * 桶的容量决定能承受多大的突发（burst），补充速率决定稳态 QPS。
 *
 * <p><b>闸二：信号量（管并发）。</b>
 * 令牌桶管不了并发占用：一个请求拿了令牌之后要跑 10 秒，
 * 期间桶里照样补令牌，很快就堆出 20 个同时在跑。
 * 而内核是有状态的单进程，靠的是「同时只处理少量请求」来保证延迟。
 * 所以并发上限必须单独设一道硬闸 —— 它才是真正保护内核的那道。
 *
 * <p>经验值：内核单次 7~12 秒，P95 约 20 秒。
 * 8 个并发已经能让 P95 明显抬头，所以 maxConcurrentStreams 默认给 8。
 * 这个数字不是拍的，是拿"内核能扛住多少并发而不显著变慢"试出来的。
 */
@Service
public class RateLimiter {

    private final GatewayProperties.RateLimit cfg;
    private final Map<String, Bucket> buckets = new ConcurrentHashMap<>();
    private final Semaphore streamSlots;

    /** 上一次做全量清理的桶数量水位，避免 ConcurrentHashMap 无限涨。 */
    private final AtomicLong lastSweep = new AtomicLong(System.currentTimeMillis());

    public RateLimiter(GatewayProperties props) {
        this.cfg = props.getRateLimit();
        this.streamSlots = new Semaphore(cfg.getMaxConcurrentStreams());
    }

    /** 令牌桶。可变状态全在 synchronized 里，保证补充与消耗是原子的。 */
    private static final class Bucket {
        double tokens;
        long lastRefillNanos;

        Bucket(double tokens, long now) {
            this.tokens = tokens;
            this.lastRefillNanos = now;
        }

        synchronized boolean tryConsume(double burst, double refillPerSecond, long now) {
            double elapsedSec = (now - lastRefillNanos) / 1_000_000_000.0;
            tokens = Math.min(burst, tokens + elapsedSec * refillPerSecond);
            lastRefillNanos = now;
            if (tokens >= 1.0) {
                tokens -= 1.0;
                return true;
            }
            return false;
        }
    }

    /**
     * 按 key 取一个令牌。
     *
     * @param key 限流维度。这里用客户端标识（会话或 IP）。
     *            按用户维度限流是相对克制的做法 ——
     *            全局桶会让一个人把所有人挤掉。
     */
    public boolean tryAcquire(String key) {
        if (!cfg.isEnabled()) {
            return true;
        }
        long now = System.nanoTime();
        Bucket b = buckets.computeIfAbsent(key, k -> new Bucket(cfg.getBurst(), now));
        sweepIfNeeded();
        return b.tryConsume(cfg.getBurst(), cfg.getRefillPerSecond(), now);
    }

    /** 尝试占用一个并发位。非阻塞：占不到就立刻返回失败（而不是排队等）。 */
    public boolean tryAcquireStreamSlot() {
        if (!cfg.isEnabled()) {
            return true;
        }
        return streamSlots.tryAcquire();
    }

    /** 释放并发位。**必须放在 finally 或 doFinally 里**，漏一次就永久少一个并发位。 */
    public void releaseStreamSlot() {
        if (!cfg.isEnabled()) {
            return;
        }
        streamSlots.release();
    }

    public int availableStreamSlots() {
        return streamSlots.availablePermits();
    }

    public int trackedKeys() {
        return buckets.size();
    }

    /** 每 5 分钟清一次闲置桶，防止按 IP 限流时被海量 IP 撑爆内存。 */
    private void sweepIfNeeded() {
        long now = System.currentTimeMillis();
        long last = lastSweep.get();
        if (now - last < 300_000) {
            return;
        }
        if (!lastSweep.compareAndSet(last, now)) {
            return;  // 另一个线程已经在清了
        }
        long cutoff = now - 600_000;
        buckets.entrySet().removeIf(e -> {
            Bucket b = e.getValue();
            synchronized (b) {
                return b.lastRefillNanos < cutoff * 1_000_000L;
            }
        });
    }
}
