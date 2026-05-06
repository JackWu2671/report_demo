"""
definitions.py — OpenAI tool schemas for agent2.

Four tools, in the order the agent should try them for a new outline request:
  1. match_outline_template    — vector search + LLM judge on pre-built templates
  2. search_outline_templates  — vector search only, returns top-N candidates (no LLM)
  3. build_outline_from_anchor — FAISS → anchor → subtree (when no template matches)
  4. modify_outline            — patch current outline via natural-language instruction
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "match_outline_template",
            "description": (
                "在预制大纲模板库中检索并由 LLM 判断最匹配的模板，决策是否可复用。"
                "status=pending_confirm 表示找到可用模板；status=not_found 表示无匹配，需改用 build_outline_from_anchor。"
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
            "name": "search_outline_templates",
            "description": (
                "仅做向量检索，返回模板库中与需求最相似的 top-N 候选模板列表（不经 LLM 判断）。"
                "用于用户想直接浏览有哪些可用模板时调用，或在 match_outline_template 结果存疑时补充参考。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回候选数量，默认 5",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_outline_from_anchor",
            "description": (
                "从知识库实时检索：FAISS向量检索 → 锚节点选择 → 子树展开 → 初始修正，生成报告大纲。"
                "仅在 match_outline_template 返回 not_found 后调用，或用户明确要求重新生成。"
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
                "仅当已存在大纲（之前成功调用过 match_outline_template 或 build_outline_from_anchor）时可用。"
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
