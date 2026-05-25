#!/usr/bin/env python3
"""
parse_scene_xlsx.py — 将场景类 xlsx 转换为 JSON

用法:
  python3 parse_scene_xlsx.py <level>

  <level>  场景层级名称，同时作为文件名前缀，例如：
             python3 parse_scene_xlsx.py 场景       # 读 场景.xlsx，写 场景.json
             python3 parse_scene_xlsx.py 子场景     # 读 子场景.xlsx，写 子场景.json

输入: expert_knowledge/<level>.xlsx
输出: expert_knowledge/<level>.json

字段说明:
  id          — 场景唯一标识
  name        — 场景名称（同 SCENEKEY）
  level       — 由命令行参数指定（如 "场景" / "子场景"）
  description — 一句话描述
  detail      — 详细拓展逻辑（原字段名 expandLogic）
  keywords    — 关键词列表（原字段名 keyWords）
  sampleIssue — 示例提问
  condition   — 触发条件（原数据无此字段，默认空字符串）
  dimensions  — 子维度列表，每项只保留 id / name / rank
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")


def resolve_paths(level: str) -> tuple[str, str]:
    input_file  = os.path.join(_KB_DIR, f"{level}.xlsx")
    output_file = os.path.join(_KB_DIR, f"{level}.json")
    return input_file, output_file


def parse_dimensions(raw) -> list[dict]:
    """只保留每个 dimension 的 id / name / rank。"""
    if not isinstance(raw, list):
        return []
    result = []
    for d in raw:
        result.append({
            "id":   d.get("id",   ""),
            "name": d.get("name", ""),
            "rank": d.get("rank", None),
        })
    # 按 rank 升序排列
    result.sort(key=lambda x: (x["rank"] is None, x["rank"]))
    return result


def parse_content(content_str: str) -> dict | None:
    """解析 CONTENT 列的 JSON 字符串，失败返回 None。"""
    try:
        return json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return None


def convert_row(scene_key: str, content_str: str, level: str) -> dict | None:
    """将一行 Excel 数据转换为目标结构。"""
    obj = parse_content(content_str)
    if obj is None:
        print(f"[警告] SCENEKEY={scene_key!r} 的 CONTENT 解析失败，跳过", file=sys.stderr)
        return None

    return {
        "id":          obj.get("id", ""),
        "name":        obj.get("name", scene_key),
        "level":       level,
        "description": obj.get("description", ""),
        "detail":      obj.get("expandLogic", ""),
        "keywords":    obj.get("keyWords", []),
        "sampleIssue": obj.get("sampleIssue", ""),
        "condition":   "",
        "dimensions":  parse_dimensions(obj.get("dimensions", [])),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python3 parse_scene_xlsx.py <level>", file=sys.stderr)
        print("示例: python3 parse_scene_xlsx.py 场景", file=sys.stderr)
        print("      python3 parse_scene_xlsx.py 子场景", file=sys.stderr)
        sys.exit(1)

    level = sys.argv[1].strip()
    input_file, output_file = resolve_paths(level)

    try:
        import openpyxl
    except ImportError:
        print("[错误] 请先安装 openpyxl: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"[错误] 文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)

    wb = openpyxl.load_workbook(input_file, data_only=True)
    ws = wb.active

    # 读取表头，找 SCENEKEY / CONTENT 列索引（大小写不敏感）
    headers = [str(cell.value).strip().upper() if cell.value else "" for cell in ws[1]]
    try:
        idx_key     = headers.index("SCENEKEY")
        idx_content = headers.index("CONTENT")
    except ValueError:
        print(f"[错误] 找不到必要列，实际表头: {headers}", file=sys.stderr)
        sys.exit(1)

    scenes = []
    for row_num, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        scene_key    = str(row[idx_key]).strip()   if row[idx_key]     else ""
        content_str  = str(row[idx_content]).strip() if row[idx_content] else ""

        if not scene_key and not content_str:
            continue  # 跳过空行

        item = convert_row(scene_key, content_str, level)
        if item:
            scenes.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(scenes)} 条 [{level}] → {output_file}")


if __name__ == "__main__":
    main()
