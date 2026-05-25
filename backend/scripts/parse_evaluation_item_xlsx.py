#!/usr/bin/env python3
"""
parse_evaluation_item_xlsx.py — 将评估项.xlsx 转换为 评估项.json

输入: expert_knowledge/评估项.xlsx
输出: expert_knowledge/评估项.json

字段说明:
  id                — 原始 UUID
  nodeId            — 短编号，LLM 大纲可见（如 L4_001）
  name              — 评估项名称
  level             — 固定 "评估项"
  description       — 一句话描述
  keywords          — 关键词列表
  sampleIssue       — 示例提问
  condition         — 整体展示条件（从 expandLogic showWhen 提取）
  summarySuggestion — LLM 总结指令
  template          — expandLogic 原文（含 ${} 占位符，执行引擎填充生成报告文字）
  metrics           — 所有关联指标名列表
"""

import json
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

# ── 配置区 ────────────────────────────────────────────────────────────
INPUT_FILE   = os.path.join(_KB_DIR, "评估项.xlsx")
OUTPUT_FILE  = os.path.join(_KB_DIR, "评估项.json")
ID_PREFIX    = "L4"   # 短 id 前缀，生成 L4_001 / L4_002 ...
ID_START     = 1      # 起始序号
# ─────────────────────────────────────────────────────────────────────


# ── id 生成 ───────────────────────────────────────────────────────────

def make_short_id(index: int) -> str:
    return f"{ID_PREFIX}_{index:03d}"


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


# ── 行转换 ────────────────────────────────────────────────────────────

def convert_row(scene_key: str, content_str: str, index: int) -> dict | None:
    try:
        obj = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        print(f"[警告] SCENEKEY={scene_key!r} CONTENT 解析失败，跳过", file=sys.stderr)
        return None

    expand_logic = obj.get("expandLogic", "")

    return {
        "id":                obj.get("id", ""),
        "nodeId":            make_short_id(index),
        "name":              obj.get("name", scene_key),
        "level":             "评估项",
        "description":       obj.get("description", ""),
        "keywords":          obj.get("keyWords") or [],
        "sampleIssue":       obj.get("sampleIssue", ""),
        "condition":         extract_condition(expand_logic),
        "summarySuggestion": obj.get("summarySuggestion") or "",
        "template":          expand_logic,
        "metrics":           obj.get("metrics") or [],
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
        item = convert_row(scene_key, content_str, ID_START + len(items))
        if item:
            items.append(item)

    has_cond = sum(1 for it in items if it["condition"])
    print(f"转换完成，共 {len(items)} 条 [评估项]")
    print(f"  有整体 condition (showWhen): {has_cond}")
    print(f"  id 范围: {make_short_id(ID_START)} ~ {make_short_id(ID_START + len(items) - 1)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"→ {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
