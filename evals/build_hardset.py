"""生成「难看」的评测集：把术语式提问改写成外行口语。

为什么需要这一套？
第一版评测集的问题全部来自书里的自测题，它们本身就是**用术语提问**的
（"「断直连」断的是什么？"）。这类问题字面就能命中原文，首轮召回就够了 ——
所以实测里自纠正循环 57 条一次都没触发，那组对照实验等于没做。

真实的用户不会这么提问。他只会说"2018 年那个把支付宝和银行之间连接切断的动作
到底是干嘛的"。这种问法里没有一个关键词能字面命中原句，
才是需要"多捞一轮"的场景。

所以这里用模型把原题改写成**禁用术语的外行问法**，gold 标注保持不变。
改完之后再跑一次对照实验，才能看出自纠正循环到底有没有用 ——
这也是一个方法论示范：**评测集选得不对，实验结论就是假的。**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citepilot.config import SETTINGS  # noqa: E402
from citepilot.llm import LLM  # noqa: E402

_SYS = """把一个支付领域的技术问题，改写成「完全不懂支付行话的业务同学」会怎么问。

硬性要求：
1. **禁止使用原问题里的专业名词**（如"断直连""备付金""三流""清算""收单""牌照"等），
   要用大白话描述那个东西"是干什么的""会发生什么"，让读者靠描述去猜。
2. 保留问题的核心意图，不能改变问的是什么。
3. 可以带上生活化的场景或举例，但不要编造事实。
4. 长度控制在 40 字以内。
5. 只输出改写后的问题本身，不要引号、不要解释、不要序号。

反例（不合格）："断直连是什么意思？" —— 还是用了术语。
正例：管理层天天说要跟银行直连，我到底为什么要绕一层？"""


def main() -> int:
    src = SETTINGS.root / "evals" / "goldenset.jsonl"
    dst = SETTINGS.root / "evals" / "goldenset_hard.jsonl"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    quiz = [i for i in items if i["source"] == "quiz"][:limit]
    llm = LLM()

    out = []
    for it in quiz:
        try:
            res = llm.complete(
                [{"role": "system", "content": _SYS}, {"role": "user", "content": it["question"]}],
                temperature=0.7, max_tokens=800,
            )
            new_q = res.text.strip().strip('"').strip("'").split("\n")[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  改写失败，跳过：{exc}")
            continue
        if not new_q or len(new_q) > 80:
            continue
        out.append({
            "qid": it["qid"].replace("quiz", "hard"),
            "question": new_q,
            "source": "hard",
            "gold_chapters": it["gold_chapters"],
            "gold_chunk": "",
            "reference": it["reference"],
            "original": it["question"],
        })
        print(f"  原：{it['question'][:44]}")
        print(f"  改：{new_q}")
        print()

    dst.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print(f"已生成 {len(out)} 条口语化问题 → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
