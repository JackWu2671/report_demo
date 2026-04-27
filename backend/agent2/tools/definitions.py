"""
definitions.py — OpenAI tool schemas for agent2.

Two tools:
  generate_outline  — build an outline from the KB (full case_workflow_2 pipeline)
  modify_outline    — patch the current outline via natural-language instruction
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_outline",
            "description": (
                "根据用户的分析需求，从知识库中检索并生成初始报告大纲。"
                "返回 Markdown 格式大纲供用户阅读。"
                "当用户首次提出分析需求、或明确要求重新生成大纲时调用。"
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
                "对当前报告大纲进行修改，支持删除章节、聚焦特定方向、设置参数阈值等。"
                "仅当已存在大纲时才可调用。"
                "返回修改后的 Markdown 格式大纲。"
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
