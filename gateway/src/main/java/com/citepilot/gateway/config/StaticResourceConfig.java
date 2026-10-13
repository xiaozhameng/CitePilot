package com.citepilot.gateway.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.config.ResourceHandlerRegistry;
import org.springframework.web.reactive.config.WebFluxConfigurer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 把内置语料目录挂到 {@code /corpus/**} 下。
 *
 * <p>这一个类解决的是一条具体的断链：引用里的锚点是
 * {@code corpus/payment-handbook/index.html#ch-3-2}，而之前**没有任何服务
 * 通过 HTTP 提供这个文件** —— 界面上每个「查看原文 →」点下去都是 404。
 * 「答案必带出处」于是只剩半句真话：角标能点，但点了没东西。
 *
 * <p>为什么用 WebFlux 的资源处理器，而不是写一个 Controller？
 * 因为静态文件的活它都替我们干好了：按扩展名推断 Content-Type、
 * 支持 Range（PDF 要跳页）、带 If-Modified-Since 的缓存协商、
 * 以及**路径穿越防护**（`/corpus/../../etc/passwd` 会被拒）。
 * 自己写一个 {@code Files.readAllBytes} 很容易，把这四件事都写对很难。
 *
 * <p>启动时就把目录解析成绝对路径并检查存在性，而不是等到第一个请求。
 * 这样配错路径的表现是**启动日志里一条醒目的 WARN**，
 * 而不是运行时某个链接莫名 404 —— 后者要排查很久。
 */
@Configuration
public class StaticResourceConfig implements WebFluxConfigurer {

    private static final Logger log = LoggerFactory.getLogger(StaticResourceConfig.class);

    private final GatewayProperties props;

    public StaticResourceConfig(GatewayProperties props) {
        this.props = props;
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        GatewayProperties.Corpus corpus = props.getCorpus();
        if (!corpus.isEnabled()) {
            log.info("语料静态托管已关闭（gateway.corpus.enabled=false），/corpus/** 由上游负责");
            return;
        }

        Path dir = Paths.get(corpus.getDir()).toAbsolutePath().normalize();
        if (!Files.isDirectory(dir)) {
            // 只告警、不抛异常：语料托管失败不该让整个网关起不来 ——
            // 问答链路并不依赖它，牺牲可用性去换一个静态目录是不划算的。
            log.warn("语料目录不存在，/corpus/** 会返回 404：{}（检查 gateway.corpus.dir）", dir);
            return;
        }

        // addResourceLocations 要的是 URL，不是文件系统路径。
        // 末尾的斜杠不能少：少了它 Spring 会当成"文件前缀"去拼，
        // 结果变成 /Users/xxx/corpusindex.html 这种路径。
        String location = dir.toUri().toString();
        if (!location.endsWith("/")) {
            location += "/";
        }

        registry.addResourceHandler("/corpus/**").addResourceLocations(location);
        log.info("语料静态托管：/corpus/**  →  {}", dir);
    }
}
