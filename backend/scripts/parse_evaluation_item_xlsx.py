#!/usr/bin/env python3
"""
parse_evaluation_item_xlsx.py — 将评估项.xlsx 转换为 评估项.json

输入: expert_knowledge/评估项.xlsx
输出: expert_knowledge/评估项.json

字段说明:
  id                — 短 id，LLM 大纲里可见（如 E001）
  uuid              — 原始 UUID
  name              — 评估项名称
  level             — 固定 "评估项"
  description       — 一句话描述
  keywords          — 关键词列表
  sampleIssue       — 示例提问
  condition         — 整体展示条件（从 expandLogic showWhen 提取）
  summarySuggestion — LLM 总结指令
  detail            — expandLogic 原文（含 ${} 模板，执行引擎填充）
  conditionMetrics  — 仅用于条件判断的指标（不展示）
  displayGroups     — 展示分组，每组含 step / title / condition / metrics
"""

import json
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

# ── 配置区 ────────────────────────────────────────────────────────────
INPUT_FILE  = os.path.join(_KB_DIR, "评估项.xlsx")
OUTPUT_FILE = os.path.join(_KB_DIR, "评估项.json")
ID_PREFIX   = "E"   # 短 id 前缀，生成 E001 / E002 ...
# ─────────────────────────────────────────────────────────────────────


# ── id 生成 ───────────────────────────────────────────────────────────

def make_short_id(index: int) -> str:
    return f"{ID_PREFIX}{index:03d}"


# ── expandLogic 解析 ──────────────────────────────────────────────────

def extract_condition(expand_logic: str) -> str:
    """
    从 expandLogic 第一行提取 showWhen 条件表达式。
    示例：## 标题.showWhen(${number("AEC覆盖用户数")=0})
    → 返回 ${number("AEC覆盖用户数")=0}
    """
    if not expand_logic:
        return ""
    first_line = expand_logic.split("\n")[0]
    m = re.search(r"\.showWhen\((.+)\)\s*$", first_line)
    return m.group(1).strip() if m else ""


# ── code 字段解析 ─────────────────────────────────────────────────────

def extract_python_code(code_field) -> str:
    """从 code 字段（codeMirror JSON）取出 Python 代码字符串。"""
    if not code_field:
        return ""
    try:
        if isinstance(code_field, str):
            code_field = json.loads(code_field)
        return code_field.get("properties", {}).get("content", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def resolve_show_content(code_expr: str, python_code: str) -> list[str]:
    """
    从 show_content(...) 表达式提取指标名列表。

    两种形式：
      1. 字面量列表  show_content(['指标A', '指标B'])
      2. 变量引用    show_content(var_name)
         → 在 python_code 中找 var_name = ['指标A', '指标B']
    """
    # 字面量列表
    m = re.search(r"show_content\s*\(\s*\[([^\]]+)\]\s*\)", code_expr)
    if m:
        return re.findall(r'["\']([^"\']+)["\']', m.group(1))

    # 变量引用
    m = re.search(r"show_content\s*\(\s*(\w+)\s*\)", code_expr)
    if m:
        var_name = m.group(1)
        vm = re.search(rf'{re.escape(var_name)}\s*=\s*\[([^\]]+)\]', python_code)
        if vm:
            return re.findall(r'["\']([^"\']+)["\']', vm.group(1))

    return []


# ── compileResultInfo 解析 ────────────────────────────────────────────

def parse_display_groups(compile_info, python_code: str) -> list[dict]:
    """
    从 compileResultInfo.graph 提取结构化 displayGroups。
    取 SHOW_CONTENT 节点，结合 stepDescriptions 拼出每组标题。
    """
    if not compile_info:
        return []
    try:
        graph = json.loads(compile_info.get("graph", "{}"))
        props = graph.get("properties", {})
        nodes = json.loads(props.get("codeDisplayNodes", "[]"))
        steps = json.loads(props.get("stepDescriptions", "[]"))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []

    # step（字符串） → 标题
    step_title = {str(s["step"]): s["description"].strip() for s in steps}

    show_nodes = sorted(
        [n for n in nodes if n.get("type") == "SHOW_CONTENT"],
        key=lambda x: x.get("step", 0),
    )

    groups = []
    for node in show_nodes:
        step_str = str(node.get("step", ""))
        title    = step_title.get(step_str, node.get("description", "")).strip()
        metrics  = resolve_show_content(node.get("code_expr", ""), python_code)

        if not metrics:
            continue

        groups.append({
            "step":      node.get("step", 0),
            "title":     title,
            "condition": "",   # 组级条件暂留空，复杂 IF 分支需人工补充
            "metrics":   metrics,
        })

    return groups


# ── 行转换 ────────────────────────────────────────────────────────────

def convert_row(scene_key: str, content_str: str, index: int) -> dict | None:
    try:
        obj = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        print(f"[警告] SCENEKEY={scene_key!r} CONTENT 解析失败，跳过", file=sys.stderr)
        return None

    all_metrics  = obj.get("metrics") or []
    expand_logic = obj.get("expandLogic", "")
    condition    = extract_condition(expand_logic)
    python_code  = extract_python_code(obj.get("code"))
    display_groups = parse_display_groups(obj.get("compileResultInfo"), python_code)

    # conditionMetrics：出现在 metrics 但不在任何 displayGroup 里的指标
    display_set      = {m for g in display_groups for m in g["metrics"]}
    condition_metrics = [m for m in all_metrics if m not in display_set]

    return {
        "id":                make_short_id(index),
        "uuid":              obj.get("id", ""),
        "name":              obj.get("name", scene_key),
        "level":             "评估项",
        "description":       obj.get("description", ""),
        "keywords":          obj.get("keyWords") or [],
        "sampleIssue":       obj.get("sampleIssue", ""),
        "condition":         condition,
        "summarySuggestion": obj.get("summarySuggestion") or "",
        "detail":            expand_logic,
        "conditionMetrics":  condition_metrics,
        "displayGroups":     display_groups,
    }


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    try:
        import openpyxl
    except ImportError:
        print("[错误] 请先安装 openpyxl: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"[错误] 文件不存在: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    wb = openpyxl.load_workbook(INPUT_FILE, data_only=True)
    ws = wb.active

    headers = [str(c.value).strip().upper() if c.value else "" for c in ws[1]]
    try:
        idx_key     = headers.index("SCENEKEY")
        idx_content = headers.index("CONTENT")
    except ValueError:
        print(f"[错误] 找不到必要列，实际表头: {headers}", file=sys.stderr)
        sys.exit(1)

    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        scene_key   = str(row[idx_key]).strip()    if row[idx_key]     else ""
        content_str = str(row[idx_content]).strip() if row[idx_content] else ""
        if not scene_key and not content_str:
            continue
        item = convert_row(scene_key, content_str, len(items) + 1)
        if item:
            items.append(item)

    # 解析统计
    no_groups = sum(1 for it in items if not it["displayGroups"])
    has_cond  = sum(1 for it in items if it["condition"])
    print(f"转换完成，共 {len(items)} 条")
    print(f"  有 displayGroups: {len(items) - no_groups}  无: {no_groups}（可能需要检查）")
    print(f"  有整体 condition: {has_cond}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"→ {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
