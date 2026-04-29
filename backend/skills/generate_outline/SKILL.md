---
name: generate_outline
description: 根据用户描述生成报告大纲，支持多轮对话式修改。
---

## 功能

接收用户的自然语言分析需求，优先从模板库检索已有大纲，无匹配时从知识库实时生成。
支持多轮对话式修改，直到用户满意。

## 入参

- `user_message: str` — 用户的自然语言输入（分析需求或修改指令）

## 产出事件（SSE）

| type | 说明 |
|------|------|
| `step` | 工具执行进度（running / done） |
| `outline` | 大纲数据（markdown + outline_tree） |
| `confirm` | 需要用户确认（使用模板 or 重新生成） |
| `text` | LLM 的简短文字说明 |
| `done` | 本轮结束，含耗时 |
| `error` | 执行异常 |

## 最终产物

调用 `skill.outline` 获取：

```python
{
    "tree": dict,       # 结构化大纲 JSON，供代码逻辑操作
    "markdown": str,    # 渲染给用户的 Markdown
}
```

## 内部实现

封装 `agent2/`，工具链：`search_outline_template` → `build_outline_from_anchor` → `modify_outline`
