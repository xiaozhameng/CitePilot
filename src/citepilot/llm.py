"""LLM 客户端：统一出口 + 分类重试 + token 记账。

为什么要包一层？
因为裸调模型 API 会遇到三类问题，每类的处理方式完全不同：
  1) 可重试的（超时 / 429 / 5xx）—— 退避重试，最多 3 次
  2) 不可重试的（400 参数错 / 401 鉴权错）—— 立即失败，重试只是浪费时间
  3) 需要降级的（配额耗尽 / 模型下线）—— 换模型或转人工
把它们混在一个 except 里是最常见的坑：对 400 重试三次，白白多等 10 秒。

同时把每次调用的 token 记下来 —— 成本数字是从这里来的，
没有记账就不可能有"单次成本 X 元"这种实践里最有说服力的数字。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .config import SETTINGS


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    ms: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# 只重试这些：超时、限流、上游故障。参数错误重试一万次也不会成功。
_RETRYABLE = ("timeout", "timed out", "rate limit", "429", "500", "502", "503", "504",
              "overloaded", "connection", "temporarily")


def _retryable(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in _RETRYABLE)


class LLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 api_base: str | None = None, timeout: float = 60.0,
                 max_retries: int = 3) -> None:
        self.model = model or SETTINGS.model
        self.api_key = api_key or SETTINGS.api_key
        self.api_base = api_base or SETTINGS.api_base
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0          # 本次进程内的调用次数，用于统计
        self.usage_in = 0
        self.usage_out = 0

    @property
    def enabled(self) -> bool:
        return SETTINGS.has_llm if self.api_key == SETTINGS.api_key else bool(self.api_key)

    def complete(self, messages: list[dict], temperature: float = 0.2,
                 max_tokens: int | None = None, json_mode: bool = False,
                 timeout: float | None = None, retries: int | None = None) -> LLMResult:
        """调一次模型。

        这里踩过一个很隐蔽的坑，值得写下来：
        deepseek-v4-flash 是**推理模型**，它先在 `reasoning_content` 里想一大段，
        想完才输出 `content`。而 reasoning 的 token **也计入 max_tokens**。
        于是 max_tokens 给小了（200）就会出现：
            content = ''，finish_reason = 'length'，reasoning_tokens = 199
        —— 不抛异常、不报错，静默返回空字符串。
        如果调用方没检查空串，就会拿到一个"成功"的空答案，
        然后在很远的地方以"格式不对"的形式爆出来。

        所以这里加了一道**预算翻倍重试**：只要发现"内容为空且被长度截断"，
        就把 max_tokens 放大再试一次。这比让调用方自己判空更可靠，
        因为调用方很容易忘记判。
        """
        import litellm

        budget = max_tokens or 800
        tmo = timeout or self.timeout
        tries = self.max_retries if retries is None else retries

        # 外层：预算重试（专门治"空内容"）
        for budget_round in range(3):
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "timeout": tmo,
                "max_tokens": budget,
                # 必须显式关掉 litellm 的内置重试！
                # 它默认 num_retries=2（即 3 次尝试），会把 timeout 放大三倍：
                # 设 timeout=5s 实测跑了 17.1s，加上自己的重试层就成了 30s+。
                # 重试策略必须只有一处 —— 这里归我们自己管（上面的内层循环），
                # 否则两个重试层相乘，延迟完全失控。
                "num_retries": 0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # 内层：错误重试（超时 / 限流 / 5xx）
            for attempt in range(tries):
                t0 = time.perf_counter()
                try:
                    resp = litellm.completion(**kwargs)
                    ms = int((time.perf_counter() - t0) * 1000)
                    usage = getattr(resp, "usage", None)
                    choice = resp.choices[0]
                    text = choice.message.content or ""
                    reasoning = 0
                    details = getattr(usage, "completion_tokens_details", None)
                    if details is not None:
                        reasoning = getattr(details, "reasoning_tokens", 0) or 0

                    self.calls += 1
                    self.usage_in += getattr(usage, "prompt_tokens", 0) or 0
                    self.usage_out += getattr(usage, "completion_tokens", 0) or 0

                    # 被长度截断且没吐出内容 → 预算不够，翻倍重来
                    if not text.strip() and getattr(choice, "finish_reason", "") == "length":
                        budget = min(budget * 4, 16000)
                        break  # 跳出错误重试，进下一轮预算重试
                    return LLMResult(
                        text=text.strip(),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        reasoning_tokens=reasoning,
                        ms=ms,
                        model=self.model,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    if not _retryable(msg) or attempt == tries - 1:
                        raise LLMError(f"模型调用失败（{self.model}）：{msg}") from exc
                    time.sleep(1.2 * (2**attempt))  # 指数退避
            else:
                continue
            continue

        raise LLMError(
            f"模型调用失败（{self.model}）：{budget_round + 1} 次尝试后仍未产出内容"
            f"（最后一次 max_tokens={budget}）"
        )

    def stream(self, messages: list[dict], temperature: float = 0.1,
               max_tokens: int | None = None, timeout: float | None = None) -> "StreamHandle":
        """流式调用，返回一个可迭代的句柄。

        为什么不写成生成器函数？
        因为生成器只能 yield 内容，调用方拿不到「这次流用了多少 token、花了多久」。
        而成本与延迟是这个项目最要紧的两个数字，必须在流结束后仍然可读。
        所以返回一个对象：迭代它拿增量，迭代结束后从它身上读用量。
        """
        return StreamHandle(
            model=self.model, api_key=self.api_key, api_base=self.api_base,
            messages=messages, temperature=temperature,
            max_tokens=max_tokens or 2000, timeout=timeout or self.timeout,
            parent=self,
        )

    def complete_json(self, messages: list[dict], **kw) -> tuple[dict, LLMResult]:
        """要求模型返回 JSON。

        注意这里做了两层容错：模型有时会把 JSON 包在 ```json 围栏里，
        有时会在前后加一句解释。直接 json.loads 会炸，所以先用正则把
        最外层的花括号内容抠出来 —— 这是"参数修正"在生产里最常见的形态。
        """
        res = self.complete(messages, json_mode=True, **kw)
        raw = res.text.strip()
        try:
            return json.loads(raw), res
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    return json.loads(m.group(0)), res
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"模型未返回合法 JSON：{raw[:200]}")


class StreamHandle:
    """一次流式调用的句柄。

    迭代它 → 拿到文本增量（str）。
    迭代结束 → 读 .prompt_tokens / .completion_tokens / .ms / .reasoning_chars。

    这里有三个坑，都是实测踩出来的：

    1) **推理模型的 reasoning_content 占满时间但不该给用户看**。
       deepseek-v4-flash 会先吐一大段 reasoning_content，再吐 content。
       如果只转发 content，用户在前 10 秒会看到一片空白，
       像是卡死了。所以 reasoning 也要往外送，但打上不同的标记
       （见 .delivered_reasoning），让 UI 显示成"思考中…"而不是正文。

    2) **usage 不一定在最后一个 chunk 里**。
       要显式传 stream_options={"include_usage": True} 才会带。
       但有的 provider 不认这个参数，直接 400。
       所以这里做了降级：带参数失败 → 去掉参数重来。

    3) **最后一个 chunk 的 choices 常常是空列表**（只带 usage）。
       直接取 choices[0] 会 IndexError —— 流式代码最经典的一行崩溃。
    """

    def __init__(self, *, model: str, api_key: str, api_base: str, messages: list[dict],
                 temperature: float, max_tokens: int, timeout: float, parent: "LLM",
                 reasoning_flush_chars: int = 28,
                 reasoning_flush_ms: int = 150) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._parent = parent
        self._rsn_flush_chars = reasoning_flush_chars
        self._rsn_flush_ms = reasoning_flush_ms

        # 流结束后可读的用量
        self.text = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_chars = 0
        self.ms = 0
        self.finished = False
        self.error: str | None = None

    def __iter__(self):
        import litellm

        base: dict = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "stream": True,
            "num_retries": 0,
        }

        t0 = time.perf_counter()
        resp = None
        # 先试带 usage 上报；上游不认就降级成不带。
        for with_usage in (True, False):
            kwargs = dict(base)
            if with_usage:
                kwargs["stream_options"] = {"include_usage": True}
            try:
                resp = litellm.completion(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if with_usage:
                    continue  # 多半是 provider 不认 stream_options，去掉重试
                self.error = f"{type(exc).__name__}: {exc}"
                self.ms = int((time.perf_counter() - t0) * 1000)
                raise LLMError(f"流式调用失败（{self.model}）：{self.error}") from exc

        # 推理草稿的攒批缓冲。
        # 为什么要攒？实测直接逐 token 转发时，一次问答能产生 150+ 个
        # 只含一两个字的 reasoning 事件（"We"、" need"、" answer"…）。
        # 每个事件都要单独编一帧 SSE，帧头开销比内容还大，
        # 前端也要为每个事件触发一次 DOM 更新 —— 纯属浪费。
        # 攒到一定长度或间隔再发，事件数能降一个数量级，内容一点不少。
        rsn_buf: list[str] = []
        rsn_last = time.perf_counter()

        def flush_reasoning() -> dict | None:
            if not rsn_buf:
                return None
            text = "".join(rsn_buf)
            rsn_buf.clear()
            return {"type": "reasoning", "text": text}

        try:
            for chunk in resp:  # type: ignore[union-attr]
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self.prompt_tokens = getattr(usage, "prompt_tokens", 0) or self.prompt_tokens
                    self.completion_tokens = (
                        getattr(usage, "completion_tokens", 0) or self.completion_tokens
                    )
                # 只带 usage 的收尾 chunk，choices 是空列表 —— 必须判空
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                # 推理草稿：不并入正文，单独计数，攒批后交给上层决定是否展示
                rsn = getattr(delta, "reasoning_content", None)
                if rsn:
                    self.reasoning_chars += len(rsn)
                    rsn_buf.append(rsn)
                    now = time.perf_counter()
                    if (sum(len(x) for x in rsn_buf) >= self._rsn_flush_chars
                            or (now - rsn_last) * 1000 >= self._rsn_flush_ms):
                        ev = flush_reasoning()
                        if ev:
                            rsn_last = now
                            yield ev

                piece = getattr(delta, "content", None)
                if piece:
                    # 正文开始输出前，先把攒着的推理草稿吐干净，
                    # 否则会出现"正文已经出来了、思考内容又滞后冒出来"的错序。
                    ev = flush_reasoning()
                    if ev:
                        yield ev
                    self.text += piece
                    yield {"type": "content", "text": piece}
        finally:
            self.ms = int((time.perf_counter() - t0) * 1000)
            self.finished = True
            # 兜底记账：有些 provider 全程不给 usage，用字符数粗估，
            # 宁可略偏也要让成本表有数（估算口径写在 meta 里，不冒充精确值）。
            if not self.prompt_tokens:
                self.prompt_tokens = sum(len(m.get("content", "")) for m in self.messages) // 2
            if not self.completion_tokens:
                self.completion_tokens = len(self.text) // 2
            self._parent.calls += 1
            self._parent.usage_in += self.prompt_tokens
            self._parent.usage_out += self.completion_tokens

    def result(self) -> LLMResult:
        """把流式结果整理成和 complete() 同构的对象，方便上层复用记账逻辑。"""
        return LLMResult(
            text=self.text.strip(),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            ms=self.ms,
            model=self.model,
        )
