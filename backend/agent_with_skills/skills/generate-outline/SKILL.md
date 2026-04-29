---
name: generate-outline
description: 根据用户描述生成报告大纲，支持多轮对话式修改。
version: 1.0.0
author: report_demo
---

## 功能

接收用户的自然语言分析需求，优先从模板库检索已有大纲，无匹配时从知识库实时生成。
支持多轮对话式修改，直到用户满意。

## 工具链

search_outline_template → build_outline_from_anchor → modify_outline

## 产出事件（SSE）

| type      | 说明                            |
|-----------|---------------------------------|
| step      | 工具执行进度（running / done）   |
| outline   | 大纲数据（markdown + tree）      |
| confirm   | 需要用户确认（使用模板 or 重新生成）|
| text      | LLM 的简短文字说明               |
| done      | 本轮结束，含耗时                 |
| error     | 执行异常                        |
