---
name: consolidate-expert
description: >
  专家知识沉淀工具包。用户发来一段较长的业务描述（通常 80～300 字），
  内容是他自己的分析判断、工作方法或场景经验，而不是在提问——这就是触发信号。
  典型表现：用陈述句描述"我们一般怎么看……"、"这个场景需要关注……"、
  "根据我的经验……"、"做这类分析要先……"，或者直接把一段业务思路一次性发过来。
  无论用户有没有说"保存"或"沉淀"，只要是在输出自己的业务知识，就必须加载此 skill。
  不适用于：用疑问句提问、请求生成报告、与业务知识分享无关的简短对话。
version: 2.0.0
author: report_demo
metadata:
  hermes:
    category: report
    tags: [expert, knowledge, template, consolidation]
---

# 沉淀专家知识

专家的业务经验分散在脑子里，无法被系统复用。这个 skill 的目标是把专家的经验提炼成
结构化模板，让后续所有报告都能站在专家的肩膀上。

## 工具调用顺序

```
search_graph_tree → set_outline_from_markdown → set_scene_metadata → [modify_outline] → save_outline_template
```

## 工作流程

### 步骤 1：检索知识库

调用 `search_graph_tree`，将专家描述的业务场景**完整原文**传入 `question`。

返回带祖先路径的节点树，包含可用的 L5 节点 id。记住这些 id，后续构造大纲时 L5 层必须引用。

- `success` → 进入步骤 2
- `not_found` → 告知专家当前知识库暂不覆盖该场景

### 步骤 2：构造大纲

根据专家输入和知识库节点，自行设计大纲结构，调用 `set_outline_from_markdown` 传入 `md_with_ids`。

`md_with_ids` 格式规则：
- 每行：`{缩进}[L{层级} {id}] {名称}（新建节点名后加全角冒号和描述）`
- 必须以唯一 L1 节点为根（报告总标题）
- L2/L3/L4 由你按专家意图自由设计，id 用 `new_001`、`new_002`… 命名
- L5 必须引用 `search_graph_tree` 返回的知识库节点 id，不可新建

调用后大纲立即展示给专家。

### 步骤 3：填写场景元数据

`set_outline_from_markdown` 调用完毕后，**立即**调用 `set_scene_metadata`，填写：
- `scene_name`：中文，不超过 10 字
- `summary`：一句话摘要，不超过 50 字
- `keywords`：3～8 个核心领域关键词
- `usage_conditions`：适用条件，不超过 80 字

### 步骤 4：按专家意见修改（按需）

调用 `modify_outline`，每次只传一个 op。修改后一句话确认变更，询问是否满意。

### 步骤 5：保存为模板

调用 `save_outline_template`。

只在专家明确确认时调用（说"保存"、"就这样"、"好的"等），不要主动催促。
保存成功后告知模板名称和存储路径。

## 注意事项

大纲通过独立事件推送给前端，**绝对不要**在文字回复里输出大纲内容或任何 Markdown 格式的结构数据。
每次工具调用后，文字回复严格控制在 1-2 句话，只说结论和下一步询问。
