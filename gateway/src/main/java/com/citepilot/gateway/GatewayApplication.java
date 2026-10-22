package com.citepilot.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import com.citepilot.gateway.config.GatewayProperties;

/**
 * CitePilot 网关。
 *
 * <p>它不实现任何问答逻辑 —— 问答逻辑全在 Python 内核里。
 * 网关负责的是内核不该管、也管不好的那部分：对外身份、流量、会话、审计。
 *
 * <p>为什么不用 Python 直接对外？
 * <ul>
 *   <li>内核是有状态的（加载了 118 条向量 + BM25 索引），扩不了副本，
 *       必须有人在它前面挡住流量；</li>
 *   <li>Python 接口面向机器（JSON + SSE），面向用户的鉴权、配额、会话
 *       不属于内核的职责，混进去会让内核越来越难测；</li>
 *   <li>内核将来会换（换模型、换检索库、甚至换语言），
 *       只要网关这条边界在，换内核不影响任何调用方。</li>
 * </ul>
 *
 * <p>这套结构对应的两块能力是「编排与运行」与「生产化」：
 * 网关是"把模型能力变成可上线服务"的那一层。
 */
@SpringBootApplication
@EnableConfigurationProperties(GatewayProperties.class)
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
