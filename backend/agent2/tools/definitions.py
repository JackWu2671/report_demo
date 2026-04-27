"""
definitions.py — OpenAI tool schemas for agent2.

Three tools, in the order the agent should try them for a new outline request:
  1. search_outline_template  — fast vector search + LLM judge on pre-built templates
  2. generate_outline         — full KB retrieval pipeline (when no template matches)
  3. modify_outline           — patch current outline via natural-language instruction
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_outline_template",
            "description": (
                "在预制大纲模板库中检索与用户需求最匹配的模板，内部自动判断是否可复用。"
                "status=found 表示找到可用模板；status=not_found 表示无匹配，需改用 generate_outline。"
                "用户提出新的分析需求时，优先调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_outline",
            "description": (
                "从知识库实时检索并生成报告大纲。"
                "仅在 search_outline_template 返回 not_found 后调用，或用户明确要求重新生成。"
                "status=not_found 表示知识库无相关内容，应告知用户系统暂不支持该场景。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": (
                "对当前报告大纲执行修改：删除章节、聚焦方向、设置参数阈值等。"
                "仅当已存在大纲（之前成功调用过 search_outline_template 或 generate_outline）时可用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "用户的自然语言修改指令，原文传入",
                    }
                },
                "required": ["instruction"],
            },
        },
    },
]
