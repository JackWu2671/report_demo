---
name: generate-outline
description: 根据用户描述生成报告大纲，支持多轮对话式修改。
version: 1.0.0
author: report_demo
---

## SOP：生成报告大纲

### 步骤 1：检索模板

调用 `search_outline_template`，传入用户需求原文。

- 返回 `status=pending_confirm` → 大纲已预览，询问用户：使用此模板，还是重新从知识库生成？
- 返回 `status=not_found` → 进入步骤 2

### 步骤 2：从知识库实时生成

调用 `build_outline_from_anchor`，传入用户需求原文。

- 返回 `status=success` → 大纲已生成，用 1-2 句话告知用户
- 返回 `status=not_found` → 告知用户系统暂不支持该场景，不要继续尝试

### 步骤 3：修改大纲（按需）

用户提出修改需求时，调用 `modify_outline`，传入自然语言修改指令。
每次修改后用 1-2 句话说明变更结果。

## 注意事项

- 大纲内容通过 outline 事件推送，**不要**在文字回复中重复输出大纲
- 每轮回复保持简短，聚焦在"做了什么"
- 已有大纲时，用户的后续输入优先理解为修改指令
