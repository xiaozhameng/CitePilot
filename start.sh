#!/usr/bin/env bash
# 一键启动 CitePilot：内核 (:8077) + 网关 (:8090)，前台阻塞。
#
# 停止方式（三种都行）：
#   · 在这个窗口按 Ctrl-C
#   · 直接关掉这个终端窗口
#   · 另开一个终端 kill 掉这个前台进程
# 两个服务会被一起收走，不会留下孤儿进程。
#
# 它会自动处理四件容易忘的事：.env 没配会生成、venv 没建会建、
# 语料切分产物缺失会重建、jar 过期会重打包。所以干净 clone 之后直接跑这一条就够。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

info() { printf "\033[36m·\033[0m %s\n" "$1"; }
warn() { printf "\033[33m!\033[0m %s\n" "$1"; }
die()  { printf "\033[31m✗\033[0m %s\n" "$1" >&2; exit 1; }

# ── 1. 配置 ────────────────────────────────────────────────
# 没有 key 也能跑：内核会自动降级成「只检索 + 原文摘录」，引用链路完全正常。
# 所以这里只提示、不退出 —— 让第一次拿到项目的人先看到东西，再决定要不要申请 key。
if [ ! -f .env ]; then
  cp .env.example .env
  warn "已从 .env.example 生成 .env"
fi

KEY="$(grep -E '^CITEPILOT_API_KEY=' .env | head -1 | cut -d= -f2- || true)"
if [ -z "$KEY" ] || [ "${KEY#sk-xxx}" != "$KEY" ]; then
  warn "CITEPILOT_API_KEY 未填 —— 本次以「原文摘录模式」启动："
  warn "  检索、引用、演示页都正常；答案不是模型写的，而是最相关的原文段落。"
  warn "  想要生成式回答，把 key 填进 $ROOT/.env 后重启即可。"
fi

# ── 2. Python 环境 ─────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
  info "首次运行，创建虚拟环境并安装依赖（约 1 分钟）…"
  python3 -m venv .venv
  .venv/bin/pip install -q -e . || die "依赖安装失败"
fi

# ── 3. 语料索引 ────────────────────────────────────────────
# chunks.jsonl 是 index.html 的派生产物，刻意不进 git（可在本地随时重建）。
# 不做这步的话，内核启动读索引时会 FileNotFoundError —— 这是干净 clone 后
# 最容易卡住的一步，所以别让它留给用户手工处理。
CHUNKS="corpus/payment-handbook/chunks.jsonl"
if [ ! -f "$CHUNKS" ]; then
  info "首次运行，切分内置语料…"
  PYTHONPATH="$ROOT/src" .venv/bin/python -m citepilot.ingest || die "语料切分失败"
fi

# ── 4. 网关 jar ────────────────────────────────────────────
# 只在 jar 缺失、或 Java 源码 / pom 比 jar 新时才打包。
JAR="gateway/target/citepilot-gateway-0.1.0.jar"
if [ ! -f "$JAR" ] || [ -n "$(find gateway/src gateway/pom.xml -newer "$JAR" -print -quit 2>/dev/null)" ]; then
  info "打包网关（首次约 1~2 分钟，Maven 要下载依赖；之后是增量）…"
  if ! ./scripts/build.sh >/tmp/citepilot-build.log 2>&1; then
    tail -15 /tmp/citepilot-build.log
    die "打包失败，完整日志：/tmp/citepilot-build.log"
  fi
fi

# ── 5. 起服务 ──────────────────────────────────────────────
echo
info "内核 :8077  ·  网关 :8090"
info "演示看板  http://127.0.0.1:8090/"
info "按 Ctrl-C 或关闭本窗口即可全部停止"
echo

exec ./scripts/dev.sh
