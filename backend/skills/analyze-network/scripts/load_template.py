#!/usr/bin/env python3
"""
按 template_id 加载模板大纲，写入会话状态。

用法:
  python3 load_template.py <template_id>

成功时输出带 id 的 Markdown 大纲。
失败时输出 JSON：{"status": "error", "message": "..."}
"""
import sys
import os
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_outline
from tools.search_template import load_template_outline


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "缺少 template_id 参数"}), file=sys.stderr)
        sys.exit(1)

    template_id = sys.argv[1]
    result = load_template_outline(template_id)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["md_with_ids"], result["markdown"])
        print(result["md_with_ids"])
    else:
        print(json.dumps({"status": "error", "message": result.get("reason", "未知错误")}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
