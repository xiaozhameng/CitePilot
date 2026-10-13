"""内核启动冒烟测试。

存在的理由（一次真实事故）：
上传接口写完、80 个测试全绿、手工 review 也过了，但 `./scripts/dev.sh`
一跑就炸 —— `python-multipart` 没装进 venv。

为什么 80 个测试都没抓到？因为它们测的都是**函数**（切分、融合、引用校验），
没有一个真去把 FastAPI 应用**装配起来**。而缺依赖这件事只在装配时才暴露：
FastAPI 是在注册路由、解析 `UploadFile = File(...)` 这个签名时
才去 ensure_multipart_is_installed()。

也就是说，之前缺的是一条"整个应用能起来吗"的检查。
这个文件就补这一条 —— 不测业务逻辑，只测「装配得起来」。

代价是毫秒级，收益是：任何一类"缺包 / 路由签名写错 / 装饰器写错"
的问题都会在这里当场报出来，而不是等你起服务的时候。
"""

from __future__ import annotations

import pytest


def test_create_app_registers_all_routes():
    """应用能装配，且关键路由都注册上了。

    这条断言的具体价值：如果哪天有人给某个 handler 的签名加了
    File/Form，却没装 python-multipart，这里会直接失败并给出安装提示，
    而不是等到跑 dev.sh 看到一长串 traceback。
    """
    from citepilot.server import create_app

    app = create_app()

    paths = set()
    for route in app.routes:
        p = getattr(route, "path", None)
        if p:
            paths.add(p)

    expected = {
        "/health",
        "/v1/ask",
        "/v1/ask/stream",
        "/v1/stats",
        "/v1/recent",
        # 知识库这一组是"动态语料"的全部入口，少任何一个都会让功能残缺：
        # 少了 GET 就看不到已传的文档，少了 DELETE 就删不掉传错的，
        # 少了 raw 则上传文档的引用点不开 —— 「答案必带出处」就成了空话。
        "/v1/kb/documents",
        "/v1/kb/documents/{doc_id}",
        "/v1/kb/documents/{doc_id}/raw",
        "/v1/kb/rebuild",
    }
    missing = expected - paths
    assert not missing, f"这些路由没注册上：{sorted(missing)}"


def test_upload_route_accepts_multipart():
    """上传路由声明的确是 multipart —— 且 body 里那个字段叫 file。

    字段名不是随便起的：网关转发时必须用同一个名字，
    否则内核收到的 body 里没有 `file`，FastAPI 直接 422。
    两边各写各的名字，是这种跨语言接缝上最容易犯的错。
    """
    from citepilot.server import create_app

    app = create_app()
    # 注意 GET 和 POST 共用同一个路径字符串，所以不能只按 path 找 ——
    # 那样会拿到先注册的 GET 路由，然后断言在一个没有 methods=POST 的对象上。
    candidates = [
        r for r in app.routes
        if getattr(r, "path", None) == "/v1/kb/documents"
        and "POST" in getattr(r, "methods", set())
    ]
    assert candidates, "找不到 POST /v1/kb/documents"
    route = candidates[0]

    # FastAPI 把解析好的依赖挂在 route.dependant 上，从中能找到 body 字段名。
    names = {
        f.name
        for f in getattr(route.dependant, "body_params", [])
    }
    assert "file" in names, f"上传字段名应为 file，实际是 {sorted(names)}"


@pytest.mark.parametrize("suffix", [".md", ".pdf", ".docx", ".html", ".txt"])
def test_every_advertised_format_is_actually_importable(suffix):
    """对外宣称支持的格式，对应的抽取器必须真的能 import。

    抽取器是懒加载的（只在真的遇到该格式时才 import），
    好处是只用 md 的人不必装 pypdf；
    代价是"依赖没装"这个问题会被推迟到用户传文件的那一刻。
    所以这里主动替每种格式探一次活。
    """
    from citepilot.kb import _EXTRACTORS, SUPPORTED_SUFFIXES

    assert suffix in SUPPORTED_SUFFIXES
    extractor = _EXTRACTORS.get(suffix)
    assert extractor is not None, f"{suffix} 宣称支持，却没有对应的抽取器"

    # 真正调用一次（空输入即可）—— 只为把懒加载的那句 import 触发出来。
    # 注意抽取器都是生成器函数：光调用不会执行函数体，
    # 必须用 list() 消费一次，里面的 import 才会真的发生。
    try:
        list(extractor(b""))
    except ImportError as e:  # pragma: no cover - 只在依赖缺失时走到
        pytest.fail(f"{suffix} 的抽取器依赖缺失：{e}")
    except Exception:
        # 空内容解析不出东西是正常的，这里只关心"模块能不能加载"。
        pass
