---
name: analyze-network
description: >
  看网分析工具包。用于一切需要分析传送网络现状的场景：覆盖评估、容量分析、
  fgOTN/OSU 部署规划、站点选址、企业覆盖缺口、资源瓶颈识别等。
  只要用户想了解网络现状、发现问题或给出部署建议，就加载此 skill——
  报告和大纲只是分析的呈现手段，不是触发条件。
  不适用于：与网络分析无关的一般性对话、简单知识问答。
version: 3.0.0
author: report_demo
metadata:
  hermes:
    category: report
    tags: [network-analysis, otn, fgotn, coverage, capacity, outline, report]
---

# 生成完整报告

用户提出分析问题，最终目标是一份完整的报告文档来回答这个问题。
报告分两个阶段生成：先确定结构（大纲），再填充内容（渲染）。

所有工具均为 Python 脚本，通过 `bash` 调用。环境变量 `$SKILLS_DIR` 已预置为脚本根目录，调用格式：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/<script>.py [参数]
```

## 脚本工具参考

| 脚本 | 说明 |
|------|------|
| `search_graph_tree.py "查询词" [--topk N] [--threshold F]` | 语义检索知识图谱节点，返回带路径的树状结构 |
| `search_templates.py "查询词" [--topk N]` | 向量检索模板库，返回候选模板 JSON 数组 |
| `build_outline.py <anchor_id>` | 以锚节点为根展开子树，生成初始大纲写入会话 |
| `modify_outline.py '<ops_json>'` | 对当前大纲执行结构化修改操作 |
| `load_template.py <template_id>` | 按 ID 加载指定模板大纲写入会话 |

## 第一阶段：生成大纲

### 步骤 1：先找现成模板

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_templates.py "用户需求原文" --topk 5
```

返回 JSON 数组，每项含 `id`、`scene_name`、`summary`、`usage_conditions`、`score`。

根据 `scene_name`、`summary`、`score` 自行判断是否有高度匹配的模板：

- **有匹配** → **不要提前加载**。仅告知用户模板名称和摘要，询问是否直接使用。  
  - 用户确认使用 → **立即调用 `load_template.py <template_id>`**，大纲将实时出现在右侧，不得用文字输出大纲内容  
  - 用户拒绝或要求重新生成 → 直接进入步骤 2，不得再次搜索模板
- **无匹配** → 直接进入步骤 2，不必告知用户"未找到模板"

### 步骤 2：从知识库实时构建

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_graph_tree.py "用户需求原文"
```

输出两部分：
1. `=== 匹配节点 ===` — 平铺列表，带路径和得分
2. `=== 相关知识树（★ 为命中节点）===` — 完整子树

- `success` → 从返回的树中按【锚节点选择原则】选定锚节点 id，进入步骤 3
- `not_found` → 知识库没有覆盖该场景，如实告知，**不要重试或编造**

#### 锚节点选择原则

从树中选择一个节点作为大纲根：
- 选与用户需求**最直接对应**的节点，而不是它的祖先节点
- 优先选有具体业务含义的叶子方向节点（L3/L4），**避免选过于宽泛的顶层节点（L1/L2）**

### 步骤 3：展开并主动修剪大纲

```bash
python3 $SKILLS_DIR/analyze-network/scripts/build_outline.py L4_001
```

输出带 id 的 Markdown 大纲，大纲同时写入会话状态并推送给前端。

生成后，**立即通过一次 `modify_outline.py` 调用完成以下两项检查**，不要等待用户指示：
1. **结构修剪**：删除与用户需求无关的节点，或用 `keep_only_node` 保留关键分支
2. **范围过滤**：若用户已指定分析范围（城市、行业、时间段、阈值等），用 `modify_node_description` 将过滤条件写入相关 query 节点描述

两项均无需操作时可不调用。修改完成后简短告知用户，询问是否需要进一步调整。

### 步骤 4：按用户反馈修改大纲（按需）

> **JSON 参数引号规则（必须遵守）**：外层用**双引号**，内层所有 `"` 转义为 `\"`。
> 不可用单引号包裹——Windows cmd.exe 不把单引号当字符串边界，参数会被空格拆散。

```bash
python3 $SKILLS_DIR/analyze-network/scripts/modify_outline.py "[{\"op\": \"delete_node\", \"node_id\": \"L4_003\"}, {\"op\": \"modify_node_description\", \"node_id\": \"L5_001\", \"value\": \"仅统计南宁市的企业\"}]"
```

支持的 op 类型：

| op | 必填字段 | 说明 |
|----|---------|------|
| `add_node` | `node_id`, `parent_id` | 从知识图谱新增节点；node_id 须来自 search_graph_tree 结果 |
| `delete_node` | `node_id` | 删除节点及其全部子树 |
| `modify_node_name` | `node_id`, `value` | 修改节点名称 |
| `modify_node_description` | `node_id`, `value` | 修改节点描述（query 节点描述决定查询范围） |
| `modify_node_condition` | `node_id`, `value` | 设置条件；格式「当……时，本节才展示」；value 传空字符串删除条件 |
| `keep_only_node` | `node_id` | 保留该节点，同级其他节点自动删除 |

**调用策略**：
- 多个独立操作合并为**一次调用**
- 若后续 op 依赖前一个 op 的结果，则**分多次调用**

成功时输出修改后的带 id 大纲。跳过的操作以 `# SKIPPED:` 开头输出——出现时**必须继续补救，不得告知用户已完成**。

### 加载模板大纲

```bash
python3 $SKILLS_DIR/analyze-network/scripts/load_template.py <template_id>
```

成功时输出带 id 的 Markdown 大纲，大纲写入会话状态并推送给前端。

## 第二阶段：渲染报告

> 渲染工具待定义，完成后补充此节。

## 浏览模板 / 知识库

- 浏览模板 → 先用 `search_templates.py` 列出候选，再用 `load_template.py` 加载具体大纲
- 浏览知识库节点 → 用 `search_graph_tree.py`

## 注意事项

- 大纲数据通过 `outline` 事件推送给前端，**绝对不要**在文字回复里输出大纲内容或任何 Markdown 格式的章节列表
- 每次脚本调用后，文字回复严格控制在 1-2 句话，只说状态结果和下一步询问
- 已有大纲时，用户的后续输入**优先理解为修改指令**，而不是新的分析请求
- 若用户只是聊天，正常回应，不调用任何脚本
