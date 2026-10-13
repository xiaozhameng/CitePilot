#!/usr/bin/env bash
# 打包 CitePilot 网关（Java / Spring Boot）成可执行 jar。
#
# 什么时候要跑：改了 gateway/ 下的 Java 代码或 static/ 下的演示页之后。
# 只是改 Python 内核的话不用跑这个 —— 内核是解释执行的。
# 产物落在 gateway/target/citepilot-gateway-0.1.0.jar，dev.sh 就是起它。
#
# 用法：
#   ./scripts/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 自己把 Java 工具链加载好，这样不要求调用方先配过 PATH。
# 注意用 if 而不是 `[ -f ] && source` —— 在 set -e 下，
# 条件为假时那个复合命令会返回非 0，脚本会当场退出。
TOOLS="$ROOT/../.tools"
if [ -f "$TOOLS/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$TOOLS/env.sh"
else
  echo "提示：未找到 $TOOLS/env.sh，将使用系统 PATH 里的 mvn" >&2
fi

if ! command -v mvn >/dev/null 2>&1; then
  echo "找不到 mvn。请先确认 $TOOLS/maven/bin/mvn 存在，或把 Maven 装好。" >&2
  exit 1
fi

cd "$ROOT/gateway"

echo "──── 打包网关（跳过 Java 测试；Python 测试走 pytest）────"
mvn -B package -DskipTests

echo
echo "✓ 产物："
ls -la "$ROOT/gateway/target/"*.jar
echo
echo "启动：./scripts/dev.sh"
