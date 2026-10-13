package com.citepilot.gateway.controller;

import com.citepilot.gateway.service.KernelClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * 知识库管理接口：上传 / 列表 / 删除 / 重建 / 取原文。
 *
 * <p><b>这一层刻意做得很薄。</b>网关只做三件事：
 * 把 multipart 原样转给内核、把内核的错误翻成调用方能看懂的响应、
 * 用超时兜住慢上传。它**不判断文件类型、不解析内容、不切分**。
 *
 * <p>理由不是"少写代码"，而是<b>避免出现第二处真相</b>：
 * 如果网关也开始认识 md / pdf / docx，那么"支持哪些格式"就有了两份实现，
 * 将来加一种格式时只改了一边，另一边会静默地拒绝 ——
 * 而且这种 bug 不会报错，只表现为"这个格式有时候能用有时候不能用"。
 * 格式知识只存在于 Python 内核里。
 *
 * <p>另一个细节：这一组接口<b>不加限流</b>。限流保护的是问答那条链路
 * （它要占用内核的并发位），而上传是运维动作，频率本来就低，
 * 给它套上 QPS 桶只会在批量导入文档时把自己卡住。
 */
@RestController
@RequestMapping("/api/kb")
public class KbController {

    private static final Logger log = LoggerFactory.getLogger(KbController.class);

    private final KernelClient kernel;
    private final ObjectMapper mapper;

    public KbController(KernelClient kernel, ObjectMapper mapper) {
        this.kernel = kernel;
        this.mapper = mapper;
    }

    /** 已入库的文档清单，以及内核支持的格式列表。 */
    @GetMapping("/documents")
    public Mono<ResponseEntity<Map<String, Object>>> list() {
        return kernel.kbList()
                .map(n -> ResponseEntity.ok(unwrap(n)))
                .onErrorResume(e -> Mono.just(toError(e, "读取知识库列表失败")));
    }

    /**
     * 上传一份文档。默认上传后立即重建索引。
     *
     * @param file    文档本体，支持 md / txt / html / pdf / docx
     * @param rebuild 是否在上传后立刻重建索引。默认 true ——
     *                否则用户传完立刻提问却召不回，会以为功能坏了。
     */
    @PostMapping(value = "/documents", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<ResponseEntity<Map<String, Object>>> upload(
            @RequestPart("file") FilePart file,
            @RequestParam(name = "rebuild", defaultValue = "true") boolean rebuild) {

        // Spring 6 的 FilePart.filename() 返回 String（5.x 时代是 Optional）
        String name = file.filename();
        if (name == null || name.isBlank()) {
            return Mono.just(ResponseEntity.badRequest()
                    .body(Map.<String, Object>of("error", "缺少文件名，无法判断文档格式")));
        }
        return kernel.kbUpload(file, rebuild)
                .map(n -> ResponseEntity.ok(unwrap(n)))
                .onErrorResume(e -> Mono.just(toError(e, "上传失败")));
    }

    /** 从知识库移除一份文档。 */
    @DeleteMapping("/documents/{docId}")
    public Mono<ResponseEntity<Map<String, Object>>> delete(
            @PathVariable String docId,
            @RequestParam(name = "rebuild", defaultValue = "true") boolean rebuild) {
        return kernel.kbDelete(docId, rebuild)
                .map(n -> ResponseEntity.ok(unwrap(n)))
                .onErrorResume(e -> Mono.just(toError(e, "删除失败")));
    }

    /** 手动重建索引。改过语料文件、或上传时跳过了重建，用它补一次。 */
    @PostMapping("/rebuild")
    public Mono<ResponseEntity<Map<String, Object>>> rebuild() {
        return kernel.kbRebuild()
                .map(n -> ResponseEntity.ok(unwrap(n)))
                .onErrorResume(e -> Mono.just(toError(e, "重建索引失败")));
    }

    /**
     * 引用深链的落地页。
     *
     * <p>上传的文档没有静态网页可跳，引用里的 anchor 指向的是这个接口。
     * 没有它，「答案必带出处」对上传文档就是一句空话 ——
     * 用户看到一个出处标签，点开却是 404。
     */
    @GetMapping("/documents/{docId}/raw")
    public Mono<ResponseEntity<Map<String, Object>>> raw(
            @PathVariable String docId,
            @RequestParam(name = "chunk", defaultValue = "") String chunk) {
        return kernel.kbRaw(docId, chunk)
                .map(n -> ResponseEntity.ok(unwrap(n)))
                .onErrorResume(e -> Mono.just(toError(e, "读取原文失败")));
    }

    // ------------------------------------------------------------------
    // 内部
    // ------------------------------------------------------------------

    /** 把内核返回的 JSON 转成 Map，直接吐给前端（网关不加自己的信封）。 */
    @SuppressWarnings("unchecked")
    private Map<String, Object> unwrap(JsonNode node) {
        if (node == null || !node.isObject()) {
            return Map.of();
        }
        return mapper.convertValue(node, Map.class);
    }

    /**
     * 把上游异常翻成调用方能看懂的响应。
     *
     * <p>这里有个关键判断：内核返回的 <b>4xx 是用户的问题</b>
     * （格式不支持、扫描版 PDF 没有文字层），必须原样透出去才 actionable；
     * <b>5xx 才是链路的锅</b>，只回一句笼统的并留下日志。
     *
     * <p>一律回 500 会怎样？用户传了个扫描版 PDF，看到"服务器内部错误"，
     * 于是开始怀疑服务坏了、重启、换文件重试 —— 而实际上只需要告诉他
     * 「这份 PDF 没有文字层，要先做 OCR」。错误信息给谁看，决定了它该多详细。
     */
    private ResponseEntity<Map<String, Object>> toError(Throwable e, String fallback) {
        if (e instanceof WebClientResponseException w) {
            String detail = detailOf(w);
            if (w.getStatusCode().is4xxClientError()) {
                return ResponseEntity.badRequest().body(Map.<String, Object>of("error", detail));
            }
            log.warn("{}：内核返回 {} {}", fallback, w.getStatusCode().value(), detail);
            return ResponseEntity.status(502)
                    .body(Map.<String, Object>of("error", "内核处理失败：" + detail));
        }
        log.warn(fallback, e);
        return ResponseEntity.status(502)
                .body(Map.<String, Object>of("error", fallback + "：" + e.getMessage()));
    }

    /** 从内核的错误响应里抠出人类可读的那句话。 */
    private String detailOf(WebClientResponseException w) {
        try {
            JsonNode n = mapper.readTree(w.getResponseBodyAsString());
            if (n.hasNonNull("detail")) {
                return n.get("detail").asText();
            }
            if (n.hasNonNull("error")) {
                return n.get("error").asText();
            }
        } catch (Exception ignored) {
            // 响应体不是 JSON（比如内核根本没起来），走下面的兜底
        }
        return "HTTP " + w.getStatusCode().value();
    }
}
