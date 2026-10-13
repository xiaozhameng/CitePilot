package com.citepilot.gateway.config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.util.concurrent.TimeUnit;

/**
 * 上游内核的 HTTP 客户端。
 *
 * <p>这里有三处配置不是随手写的，值得单独说明：
 *
 * <p><b>1) 连接池上限必须有。</b>
 * 默认的无上限连接池在遇到上游变慢时会疯狂开新连接，
 * 把上游压得更死（也叫"重试风暴"的近亲）。内核是单进程，
 * 给它 64 个连接已经是上限，再多也没有吞吐可加。
 *
 * <p><b>2) 响应头超时和响应体超时不是一回事。</b>
 * {@code responseTimeout} 管的是"从发出请求到收到响应头"。
 * 对 SSE 来说响应头秒回，所以它不会误伤长连接的流；
 * 如果错误地按"整个响应"设超时，流式会被无辜掐断。
 *
 * <p><b>3) 缓冲区要放大。</b>
 * 内核一次返回的完整答案 JSON 大约 20~40KB（含 8 条命中的预览），
 * 而 WebClient 默认的缓冲上限是 256KB —— 目前够用，
 * 但如果命中数调大就会撞线并抛出 DataBufferLimitException。
 * 提前放到 2MB，把这个坑摁死在配置阶段。
 */
@Configuration
public class WebClientConfig {

    @Bean
    public WebClient kernelWebClient(GatewayProperties props) {
        var k = props.getKernel();

        ConnectionProvider pool = ConnectionProvider.builder("kernel")
                .maxConnections(k.getMaxConnections())
                .pendingAcquireTimeout(java.time.Duration.ofSeconds(10))
                .maxIdleTime(java.time.Duration.ofSeconds(30))
                .build();

        HttpClient httpClient = HttpClient.create(pool)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, (int) k.getConnectTimeout().toMillis())
                // 给每条连接加一个读超时：内核卡死时不要无限挂着
                .doOnConnected(conn -> conn.addHandlerLast(new ReadTimeoutHandler(
                        k.getReadTimeout().toSeconds(), TimeUnit.SECONDS)))
                .responseTimeout(k.getReadTimeout());

        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(2 * 1024 * 1024))
                .build();

        return WebClient.builder()
                .baseUrl(k.getBaseUrl())
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .exchangeStrategies(strategies)
                .defaultHeader("X-Internal-Token", k.getToken())
                .build();
    }
}
