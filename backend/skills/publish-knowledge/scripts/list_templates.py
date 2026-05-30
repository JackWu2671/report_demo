#!/usr/bin/env python3
"""
列出所有已保存的专家模板。

用法:
  python3 list_templates.py [--with-outline]

默认只输出元数据（id, scene_name, summary, usage_conditions, created_at）。
加 --with-outline 后额外返回每个模板的完整大纲（outline_yaml 格式）。

输出 JSON 数组，无模板时输出 []。
"""
import sys
import os
import json
import glob
import argparse

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPTS)))
sys.path.insert(0, _BACKEND_DIR)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from outline_utils import to_yaml

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-outline", action="store_true",
                        help="在结果里附上每个模板的完整大纲（outline_yaml 格式）")
    args = parser.parse_args()

    if not os.path.isdir(_TEMPLATE_DIR):
        print("[]")
        return

    results = []
    for path in sorted(glob.glob(os.path.join(_TEMPLATE_DIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                t = json.load(f)
        except Exception:
            continue

        item = {
            "id": t.get("id", ""),
            "scene_name": t.get("scene_name", ""),
            "summary": t.get("summary", ""),
            "usage_conditions": t.get("usage_conditions", ""),
            "created_at": t.get("created_at", ""),
        }
        if args.with_outline:
            outline = t.get("outline", {})
            item["outline_yaml"] = to_yaml(outline) if outline else ""

        results.append(item)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
