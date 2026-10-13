package com.citepilot.gateway.service;

import com.citepilot.gateway.config.GatewayProperties;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 会话存储。
 *
 * <p><b>先说清楚它现在做什么、不做什么 —— 这一点比代码本身重要。</b>
 *
 * <p>它现在做三件事：
 * <ol>
 *   <li>分配与维护 sessionId，让同一轮对话的多次问答能归属到同一个会话；</li>
 *   <li>记录每一轮的问答摘要，供演示页展示"这个会话问过什么"；</li>
 *   <li>作为限流的维度（按会话限流比全局限流公平）。</li>
 * </ol>
 *
 * <p>它<b>还没有</b>做多轮上下文改写（把"它呢？"补全成"备付金的监管要求呢？"）。
 * 这件事看起来简单，做对却要额外两步：先判断本轮是否依赖上文
 * （这本身是一次模型调用，又是一份延迟与成本），
 * 再把历史按 token 预算裁剪后注入。这两步都还没做，
 * 所以这里不提供任何"看起来像是做了"的接口 ——
 * 缺什么就说什么，比放一个空实现更容易在实践里站得住。
 *
 * <p>实现上是内存态 + LRU 淘汰。它天然不支持多实例部署，
 * 换成 Redis 只需要替换这个类 —— 接口没变。
 */
@Service
public class SessionStore {

    /** 一轮问答的摘要。只存必要字段：完整答案太大，也没必要在网关留一份。 */
    public record Turn(
            String question,
            String answerPreview,
            String route,
            boolean grounded,
            int latencyMs,
            long ts
    ) {}

    /** 一个会话。用 Deque 而不是 List，是因为"只保留最近 N 轮"要频繁从头部丢。 */
    public static final class Session {
        private final String id;
        private final long createdAt;
        private volatile long lastActiveAt;
        private final Deque<Turn> turns = new ArrayDeque<>();

        Session(String id) {
            this.id = id;
            this.createdAt = System.currentTimeMillis();
            this.lastActiveAt = this.createdAt;
        }

        public String id() { return id; }
        public long createdAt() { return createdAt; }
        public long lastActiveAt() { return lastActiveAt; }

        public synchronized void addTurn(Turn t, int maxTurns) {
            turns.addLast(t);
            while (turns.size() > maxTurns) {
                turns.removeFirst();
            }
            lastActiveAt = System.currentTimeMillis();
        }

        public synchronized List<Turn> turns() {
            return new ArrayList<>(turns);
        }

        public synchronized int turnCount() {
            return turns.size();
        }
    }

    private final Map<String, Session> sessions = new ConcurrentHashMap<>();
    private final GatewayProperties.Session cfg;

    public SessionStore(GatewayProperties props) {
        this.cfg = props.getSession();
    }

    /** 复用传入的 id，或新建一个。
     *
     * <p>过期的旧会话已经在 {@link #evictIfNeeded()} 里被清掉了，
     * 所以这里用一个 computeIfAbsent 就够了 ——
     * 同名会话如果已经过期，是会先被移除、再被新建的，语义正好是"重开"。
     */
    public Session getOrCreate(String sessionId) {
        evictIfNeeded();
        String id = (sessionId != null && !sessionId.isBlank())
                ? sessionId
                : UUID.randomUUID().toString().replace("-", "").substring(0, 16);
        return sessions.computeIfAbsent(id, Session::new);
    }

    public void record(String sessionId, Turn turn) {
        if (sessionId == null || sessionId.isBlank()) {
            return;
        }
        Session s = sessions.get(sessionId);
        if (s != null) {
            s.addTurn(turn, cfg.getMaxTurns());
        }
    }

    public List<Session> list() {
        List<Session> out = new ArrayList<>(sessions.values());
        out.sort(Comparator.comparingLong(Session::lastActiveAt).reversed());
        return out;
    }

    public Map<String, Object> describe(Session s) {
        return Map.of(
                "sessionId", s.id(),
                "turns", s.turnCount(),
                "createdAt", Instant.ofEpochMilli(s.createdAt()).toString(),
                "lastActiveAt", Instant.ofEpochMilli(s.lastActiveAt()).toString()
        );
    }

    private boolean isExpired(Session s) {
        return System.currentTimeMillis() - s.lastActiveAt() > cfg.getIdleTimeout().toMillis();
    }

    /** 先清过期的，再按 LRU 淘汰到上限以内。顺序不能反，否则会把活跃会话误杀。 */
    private void evictIfNeeded() {
        sessions.entrySet().removeIf(e -> isExpired(e.getValue()));
        if (sessions.size() <= cfg.getMaxSessions()) {
            return;
        }
        List<Session> byAge = new ArrayList<>(sessions.values());
        byAge.sort(Comparator.comparingLong(Session::lastActiveAt));
        int toRemove = sessions.size() - cfg.getMaxSessions();
        for (int i = 0; i < toRemove && i < byAge.size(); i++) {
            sessions.remove(byAge.get(i).id());
        }
    }

    public int size() {
        return sessions.size();
    }
}
