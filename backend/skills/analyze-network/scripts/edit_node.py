#!/usr/bin/env python3
"""
edit_node.py — 对 L5/L1-L4 节点的某个字段做「查找替换」式修改。

为什么用替换而不是整段传值：
  exec_sql / name 这类文本里常含反引号 `…`、`<`、`>`、`%`、引号。
  无论怎么转义，这些字符过后端 shell（尤其 Windows cmd.exe）都会被吃掉：
    - 反引号 → bash 当命令替换执行
    - `<` `>` → cmd.exe 当重定向（报「系统找不到指定的文件」）
    - heredoc(`<<`) → cmd.exe 不支持（报「此时不应有 <<」）
  本脚本只让「改动的小片段」(--old / --new) 上命令行，完整文本始终留在
  会话文件里、永不过 shell。多数微调（改阈值、改档位、改过滤值）的 delta
  token（如 0.7 / 80%）天然不含上述危险字符，因此跨 shell 稳定。

用法：
  python3 edit_node.py <node_id> <field> --old <旧串> --new <新串> [--old .. --new ..]

  field ∈ { exec_sql | name | description | condition }

示例（OLT 槽位利用率阈值 70% 改 80%）：
  python3 edit_node.py L5_071 exec_sql --old 0.7 --new 0.8
  python3 edit_node.py L5_071 name     --old 70% --new 80%

匹配规则：
  - 每个 --old 都在「当前字段值」里查找；任一未命中 → 报错并回显当前值，
    不写入任何改动（绝不静默成功）。
  - 命中多处 → 全部替换，并提示各替换了几处。
  - 所有 --old 都针对「替换前的原始当前值」判断命中，再按给定顺序依次替换。

成功时输出修改后的 YAML 大纲；跳过的底层操作以 # SKIPPED: 开头。
失败时输出 JSON 错误并以非 0 退出。
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import get_outline_tree, set_outline
from modify_outline import modify_outline

# field → 底层 patch op
_FIELD_OP = {
    "exec_sql":    "modify_node_exec_sql",
    "name":        "modify_node_name",
    "description": "modify_node_description",
    "condition":   "modify_node_condition",
}


def _err(message: str, current_value: str | None = None) -> None:
    payload = {"status": "error", "message": message}
    if current_value is not None:
        payload["current_value"] = current_value
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    sys.exit(1)


def _find_node(node: dict, node_id: str) -> dict | None:
    if node.get("id") == node_id:
        return node
    for child in node.get("children", []):
        hit = _find_node(child, node_id)
        if hit is not None:
            return hit
    return None


def _parse_pairs(args: list[str]) -> list[tuple[str, str]]:
    """把 --old A --new B --old C --new D 解析成 [(A,B),(C,D)]，按出现顺序配对。"""
    olds: list[str] = []
    news: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--old":
            if i + 1 >= len(args):
                _err("--old 后缺少值")
            olds.append(args[i + 1])
            i += 2
        elif token == "--new":
            if i + 1 >= len(args):
                _err("--new 后缺少值")
            news.append(args[i + 1])
            i += 2
        else:
            _err(f"无法识别的参数：{token!r}（只接受 --old / --new）")
    if not olds:
        _err("至少需要一组 --old / --new")
    if len(olds) != len(news):
        _err(f"--old 与 --new 数量不一致：{len(olds)} 个 --old，{len(news)} 个 --new")
    return list(zip(olds, news))


async def main():
    if len(sys.argv) < 3:
        _err("用法: edit_node.py <node_id> <field> --old <旧串> --new <新串> [...]")

    node_id = sys.argv[1].strip()
    field = sys.argv[2].strip()
    if field not in _FIELD_OP:
        _err(f"不支持的字段 {field!r}，只能是: {', '.join(_FIELD_OP)}")

    pairs = _parse_pairs(sys.argv[3:])

    outline_tree = get_outline_tree()
    if not outline_tree:
        _err("当前没有大纲，请先调用 build_outline 或 load_template")

    node = _find_node(outline_tree, node_id)
    if node is None:
        _err(f"节点 {node_id} 不在当前大纲中")

    original = node.get(field, "")
    if not isinstance(original, str):
        _err(f"节点 {node_id} 的字段 {field!r} 当前不是文本（值为 {type(original).__name__}），无法替换")

    # 1) 先全部针对原始值校验命中，任一未命中即整体失败
    counts = [(old, new, original.count(old)) for old, new in pairs]
    missing = [old for old, _new, c in counts if c == 0]
    if missing:
        _err(
            f"以下 --old 未在节点 {node_id} 的 {field} 中找到，未做任何修改：{missing}",
            current_value=original,
        )

    # 2) 依次替换（每个 old 替换全部出现）
    new_value = original
    for old, new in pairs:
        new_value = new_value.replace(old, new)

    if new_value == original:
        _err(
            f"替换后内容与原值相同（--old 与 --new 可能一致），未产生改动",
            current_value=original,
        )

    ops = [{"op": _FIELD_OP[field], "node_id": node_id, "value": new_value}]
    result = await modify_outline(ops, outline_tree)

    if result["status"] != "success":
        _err(result["message"])

    set_outline(result["outline_tree"], result["outline_yaml"], result["markdown"])
    print(result["outline_yaml"])

    for old, new, c in counts:
        print(f"# REPLACED: {old!r} → {new!r}（{c} 处）")
    for s in result.get("skipped", []):
        reason = s.get("_skip_reason", "未知原因") if isinstance(s, dict) else str(s)
        op = s.get("op", "?") if isinstance(s, dict) else "?"
        nid = s.get("node_id", "") if isinstance(s, dict) else ""
        print(f"# SKIPPED: {op} node_id={nid} → {reason}")


if __name__ == "__main__":
    asyncio.run(main())
