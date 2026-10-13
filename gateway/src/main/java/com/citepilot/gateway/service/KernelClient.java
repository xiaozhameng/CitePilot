package com.citepilot.gateway.service;

import com.citepilot.gateway.config.GatewayProperties;
import com.citepilot.gateway.model.AskRequest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 访问 Python 内核。整个网关只有这一个类会真的发 HTTP 出去。
 *
 * <p>集中在一处的理由：一旦内核的协议要改（加字段、改事件名），
 * 只需要动这个文件。散在各处的话，改协议就变成了全局搜索替换。
 */
@Service
public class KernelClient {

    private static final Logger log = LoggerFactory.getLogger(KernelClient.class);

    /**
     * 注意这个类型。**这是本项目里最容易被写错的一行。**
     *
     * <p>如果写成 {@code bodyToFlux(String.class)}，
     * 拿到的是每个事件的 data 字符串，而 {@code event: xxx} 里的
     * "xxx" 会被直接丢弃 —— 因为 String 类型没有地方放事件名。
     * 结果是流式能"跑通"，但前端分不清哪个是 delta、哪个是 sources，
     * 且这个 bug 不会抛任何异常，只在 UI 上表现为"内容乱成一团"。
     *
     * <p>要保住事件名，就必须声明成 {@code ServerSentEvent<T>}。
     */
    private static final ParameterizedTypeReference<ServerSentEvent<String>> SSE_TYPE =
            new ParameterizedTypeReference<>() {};

    /**
     * 上传加上重建索引的耗时上限。
     *
     * <p>为什么给得这么宽？因为内核在上传之后会**全量重建索引**，
     * 其中包含对新块的 embedding 调用 —— 一份几百页的 PDF 能切出上千块，
     * 逐个算向量要几十秒。这个超时只是兜底，正常上传在几秒内返回。
     */
    private static final Duration UPLOAD_TIMEOUT = Duration.ofSeconds(180);

    private final WebClient client;
    private final ObjectMapper mapper;
    private final Duration idleTimeout;

    public KernelClient(WebClient kernelWebClient, ObjectMapper mapper, GatewayProperties props) {
        this.client = kernelWebClient;
        this.mapper = mapper;
        this.idleTimeout = props.getKernel().getStreamIdleTimeout();
    }

    private Map<String, Object> body(AskRequest req) {
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("question", req.question());
        b.put("strategy", req.strategyOrDefault());
        b.put("use_rerank", req.useRerankOrDefault());
        if (req.sessionId() != null && !req.sessionId().isBlank()) {
            b.put("session_id", req.sessionId());
        }
        return b;
    }

    /** 非流式：等内核算完一次性拿回完整答案。 */
    public Mono<JsonNode> ask(AskRequest req) {
        return client.post()
                .uri("/v1/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body(req))
                .retrieve()
                .bodyToMono(JsonNode.class);
    }

    /** 聚合指标（透传内核的 /v1/stats）。 */
    public Mono<JsonNode> stats() {
        return client.get().uri("/v1/stats").retrieve().bodyToMono(JsonNode.class);
    }

    /** 最近问答（透传内核的 /v1/recent）。 */
    public Mono<JsonNode> recent(int limit) {
        return client.get().uri("/v1/recent?limit={n}", limit).retrieve().bodyToMono(JsonNode.class);
    }

    /** 内核健康状态。 */
    public Mono<JsonNode> health() {
        return client.get().uri("/health").retrieve().bodyToMono(JsonNode.class);
    }

    // ------------------------------------------------------------------
    // 知识库。网关在这一层只做转发，不解析任何文档格式 ——
    // "支持哪些格式"这个知识只应该存在于内核里，存在两处必然会有一天对不上。
    // ------------------------------------------------------------------

    /** 已入库的文档清单。 */
    public Mono<JsonNode> kbList() {
        return client.get().uri("/v1/kb/documents").retrieve().bodyToMono(JsonNode.class);
    }

    /**
     * 上传一份文档。
     *
     * <p>用 {@code asyncPart} 而不是 {@code part}：前者把 {@code FilePart} 的
     * content 流直接接到出站请求上，整份文件不会被读进内存再写出。
     * 换成 {@code part(...)} 就需要先把内容收集成 byte[]，
     * 在传 PDF 那种体积上会直接把堆撑大。
     */
    public Mono<JsonNode> kbUpload(FilePart file, boolean rebuild) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.asyncPart("file", file.content(), DataBuffer.class)
                .filename(file.filename());

        return client.post()
                .uri(uriBuilder -> uriBuilder.path("/v1/kb/documents")
                        .queryParam("rebuild", rebuild).build())
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(builder.build()))
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(UPLOAD_TIMEOUT);
    }

    /** 移除一份文档。 */
    public Mono<JsonNode> kbDelete(String docId, boolean rebuild) {
        return client.delete()
                .uri(uriBuilder -> uriBuilder.path("/v1/kb/documents/{id}")
                        .queryParam("rebuild", rebuild).build(docId))
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(UPLOAD_TIMEOUT);
    }

    /** 手动重建索引。 */
    public Mono<JsonNode> kbRebuild() {
        return client.post()
                .uri("/v1/kb/rebuild")
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(UPLOAD_TIMEOUT);
    }

    /** 某一块的原文（引用深链的落地页）。 */
    public Mono<JsonNode> kbRaw(String docId, String chunkId) {
        return client.get()
                .uri(uriBuilder -> uriBuilder.path("/v1/kb/documents/{id}/raw")
                        .queryParam("chunk", chunkId).build(docId))
                .retrieve()
                .bodyToMono(JsonNode.class);
    }

    /**
     * 流式：把内核的 SSE 事件原样转出来。
     *
     * <p>三个必须处理的问题：
     *
     * <p><b>① 上游错误不能"顺着流抛出去"。</b>
     * 如果在流已经开始之后抛异常，下游收到的是一个断掉的连接 ——
     * 前端只能显示"网络错误"，完全不知道发生了什么。
     * 所以这里把异常转成一个 error 事件，让流优雅收尾。
     *
     * <p><b>② 空闲超时。</b>
     * 内核正常时每几百毫秒就有一个事件。如果超过 idleTimeout 什么都没来，
     * 说明它卡死了，必须主动断掉，否则连接会一直挂着。
     *
     * <p><b>③ 空 data 要跳过。</b>
     * 有些 SSE 实现会发纯粹的注释行（: ping）当心跳，
     * 解码出来是 data 为 null 或空串的事件。直接往下传会让前端报 JSON 解析错。
     */
    public Flux<ServerSentEvent<String>> stream(AskRequest req) {
        return client.post()
                .uri("/v1/ask/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .bodyValue(body(req))
                .retrieve()
                .bodyToFlux(SSE_TYPE)
                .filter(e -> e.data() != null && !e.data().isBlank())
                .timeout(idleTimeout)
                .onErrorResume(ex -> {
                    log.warn("内核流式中断：{}", ex.toString());
                    String msg = switch (ex) {
                        case java.util.concurrent.TimeoutException ignored ->
                                "上游超时：内核超过 " + idleTimeout.toSeconds() + " 秒没有响应";
                        default -> "上游异常：" + ex.getClass().getSimpleName() + " - " + ex.getMessage();
                    };
                    try {
                        String json = mapper.writeValueAsString(Map.of("message", msg));
                        return Flux.just(ServerSentEvent.<String>builder()
                                .event("error").data(json).build());
                    } catch (Exception e) {
                        return Flux.just(ServerSentEvent.<String>builder()
                                .event("error").data("{\"message\":\"上游异常\"}").build());
                    }
                });
    }
}
