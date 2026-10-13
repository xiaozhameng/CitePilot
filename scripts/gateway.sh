#!/usr/bin/env bash
# 启动 CitePilot 网关（Java / Spring Boot）。
#
# 网关做的事：SSE 流式透传、限流、会话、审计。
# 它不实现任何问答逻辑 —— 那些全在内核里。
#
# 用法：
#   ./scripts/gateway.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 本机 Java 工具链放在 Agent/.tools 下（免安装 tar 包解出来的，不依赖 brew）。
# source 它会设好 JAVA_HOME 与 PATH。没有这个目录就退回用系统已有的 java。
TOOLS="$ROOT/../.tools"
if [ -f "$TOOLS/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$TOOLS/env.sh"
else
  echo "提示：未找到 $TOOLS/env.sh，将使用系统 PATH 里的 java" >&2
fi

JAR="$ROOT/gateway/target/citepilot-gateway-0.1.0.jar"
if [ ! -f "$JAR" ]; then
  echo "还没有打包。先执行：" >&2
  echo "  $ROOT/scripts/build.sh" >&2
  echo "（这个脚本会自己加载 Java 工具链，不需要你先配 PATH）" >&2
  exit 1
fi

cd "$ROOT/gateway"
PORT="${GATEWAY_PORT:-8090}"

echo "CitePilot 网关 → http://127.0.0.1:${PORT}"
echo "  演示页    http://127.0.0.1:${PORT}/"
echo "  健康检查  http://127.0.0.1:${PORT}/api/health"
echo

# 显式传 --server.port：命令行参数的优先级高于任何环境变量。
# 这一步不是多此一举 —— 某些环境会注入 SERVER__PORT 之类的变量，
# 而 Spring Boot 的宽松绑定会把它当成 server.port，
# 结果是服务起在一个随机端口上，日志里还写着"started on port xxxxx"，
# 排查起来很费劲。
exec java -jar "$JAR" --server.port="$PORT"
