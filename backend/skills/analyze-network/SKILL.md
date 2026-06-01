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

## 知识体系结构

知识库按五级层次组织：**场景（L1）→ 子场景（L2）→ 评估维度（L3）→ 评估项（L4）→ 评估指标（L5/Q）**。

**L5 query 节点的特殊性**：
- `name` 就是查询语句本身（如"10GPON套餐用户占比"），是指标的唯一标识，系统据此执行 SQL
- `description` **永远为空**，没有任何含义，禁止写入
- 如需修改 L5 节点，**只能改 `name`**，且新 name 须对应知识库中真实存在的指标
- **改名 = 换指标**：`modify_node_name` 会自动将节点的 `exec_sql`、`renderType`、`colX`、`colY` 等字段全部替换为新指标的值，相当于整体换掉这条查询

**修改 L5 节点前必须先查详情**：

用户表达修改 L5 节点的意图时，**必须先调用 `get_node_detail.py`** 查看当前节点的 `exec_sql`，确认当前节点在查什么数据，再结合用户需求判断是否需要换指标以及换成哪个。

```bash
python3 $SKILLS_DIR/analyze-network/scripts/get_node_detail.py L5_001
```

查到详情后：
1. 若当前指标已符合用户需求 → 无需修改，直接告知用户
2. 若需要换成其他指标 → 用 `search_graph_tree.py` 找到目标指标的节点 id，再调 `modify_node_name`

## 脚本工具参考

| 脚本 | 说明 |
|------|------|
| `search_graph_tree.py "查询词" [--topk N] [--threshold F]` | 语义检索知识图谱节点，返回带路径的树状结构 |
| `search_templates.py "查询词" [--topk N]` | 向量检索模板库，返回候选模板 JSON 数组 |
| `build_outline.py <anchor_id>` | 以锚节点为根展开子树，生成初始大纲写入会话 |
| `modify_outline.py '<ops_json>'` | 对当前大纲执行结构化修改操作 |
| *(原生工具)* `edit_node` | **修改节点属性值**（exec_sql/name/description/condition 等）—— 直接用对话工具调用，不是脚本。参数走 JSON、不过 shell，含反引号/`<`/`>` 均安全 |
| `load_template.py <template_id>` | 按 ID 加载指定模板大纲写入会话 |
| `get_node_detail.py <node_id> [node_id2 ...]` | 查询节点完整信息（summarySuggestion、exec_sql、renderType 等） |
| `get_report_data.py <node_id>` | 查询已生成报告中某节点的指标数据（每项前 10 行）和总结文本 |

## 第一阶段：生成大纲

### 步骤 1：先找现成模板

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_templates.py "用户需求原文" --topk 5
```

返回 JSON 数组，每项含 `id`、`scene_name`、`summary`、`usage_conditions`、`score`。

根据 `scene_name`、`summary`、`score` 自行判断是否有高度匹配的模板：

- **有匹配** → 立即调用 `load_template.py <template_id>`，大纲将实时出现在右侧。  
  加载完成后，告知用户找到了哪个模板，并提供以下两个选项供用户选择：  
  - 「使用此模板」  
  - 「重新生成」  
  **等待用户选择，不得自行决定。**  
  - 用户选择「使用此模板」→ 进入步骤 4，询问是否需要进一步修改  
  - 用户选择「重新生成」→ 执行步骤 2，不得再次搜索模板
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

输出 YAML 格式大纲，大纲同时写入会话状态并推送给前端。

生成后，**立即通过一次 `modify_outline.py` 调用完成结构修剪**，不要等待用户指示：
- 删除与用户需求无关的节点，或用 `keep_only_node` 保留关键分支
- 无需修剪时可不调用

修改完成后，用一句话告知用户大纲已生成，并询问：「是否需要调整大纲？如果满意，我可以直接生成报告。」

### 步骤 4：按用户反馈修改大纲（按需）

**两种调用方式：**

**参数模式**（普通操作，value 不含反引号且不含单引号）：
> 外层用**双引号**，内层所有 `"` 转义为 `\"`。不可用单引号——Windows cmd.exe 不把单引号当字符串边界。

```bash
python3 $SKILLS_DIR/analyze-network/scripts/modify_outline.py "[{\"op\": \"delete_node\", \"node_id\": \"L4_003\"}, {\"op\": \"modify_node_name\", \"node_id\": \"L4_007\", \"value\": \"新名称\"}]"
```

**修改节点属性值（exec_sql / name / description / condition 等）→ 用原生 `edit_node` 工具**：

> **禁止**把含反引号 `` ` ``、`<`、`>` 的完整 SQL 或名称当 bash 参数传给 `modify_outline.py`：
> cmd.exe 会把 `>=70%` 的 `>` 当重定向、把反引号当命令替换，导致静默失败。
>
> 正确做法：调用 `edit_node` 工具（原生 JSON 工具调用，不走 shell）。

```
edit_node(node_id="L5_071", field="exec_sql", value="SELECT ... `档位` ...")
edit_node(node_id="L5_071", field="name",     value="OLT槽位利用率分布（>=80%）")
```

支持的 field：`exec_sql` / `name` / `description` / `condition` / `summarySuggestion` / `renderType` / `colX` / `colY` / `condition_queries`  
修改成功后自动推送 `outline` 事件，前端三个 Tab 同步更新。

支持的 op 类型：

| op | 必填字段 | 可选字段 | 说明 |
|----|---------|---------|------|
| `add_node` | `node_id`, `parent_id` | `after_id` | 从知识图谱新增节点；node_id 须来自 search_graph_tree 结果 |
| `delete_node` | `node_id` | — | 删除节点及其全部子树 |
| `modify_node_name` | `node_id`, `value` | — | 修改节点名称；对 L5 节点会自动同步 exec_sql 等所有关联字段，**调用前必须先用 `get_node_detail.py` 查清楚当前节点** |
| `modify_node_description` | `node_id`, `value` | — | 修改节点描述（**仅限 L1–L4**；L5 query 节点无 description，操作会被拒绝） |
| `modify_node_condition` | `node_id`, `value` | — | 设置条件；格式「当……时，本节才展示」；value 传空字符串删除条件 |
| `modify_node_exec_sql` | `node_id`, `value` | — | 直接修改 L5 节点的 `exec_sql`（仅限 L5）。**含反引号/`<`/`>` 时禁止走 bash，改用 `edit_node` 工具** |
| `keep_only_node` | `node_id` | — | 保留该节点，同级其他节点自动删除 |

**`add_node` 位置规则（重要）**：

- `parent_id` = 新节点的**父节点** ID，**绝对不能**填兄弟节点的 ID
- `after_id`（可选）= 新节点插入到该兄弟节点**之后**；省略时追加到父节点末尾
- **新增同级节点**示例：在 L4_012 之后插入 L4_013
  ```json
  {"op":"add_node","node_id":"L4_013","parent_id":"<L4_012的父节点ID>","after_id":"L4_012"}
  ```
  ❌ 错误：`"parent_id":"L4_012"` → L4_013 会变成 L4_012 的子节点

**调用策略**：
- 多个独立操作合并为**一次调用**
- 若后续 op 依赖前一个 op 的结果，则**分多次调用**

成功时输出修改后的 YAML 大纲。跳过的操作以 `# SKIPPED:` 开头输出——出现时**必须继续补救，不得告知用户已完成**。

### 加载模板大纲

```bash
python3 $SKILLS_DIR/analyze-network/scripts/load_template.py <template_id>
```

成功时输出 YAML 格式大纲，大纲写入会话状态并推送给前端。

## 第二阶段：生成报告

当用户确认大纲满意时（说「可以了」「开始生成」「生成报告」「就这样」「ok」等），调用：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/trigger_report.py
```

触发后回复一句「好的，开始生成报告。」，不需要等待报告完成，前端会自动处理渲染。

**识别意图的原则**：
- 用户说修改意图 → 步骤 4，修改大纲
- 用户表达确认/满意/开始 → 触发报告生成
- 两者同时出现（如「把xx改一下然后生成」）→ 先修改大纲，再触发

## 浏览模板 / 知识库

- 浏览模板 → 先用 `search_templates.py` 列出候选，再用 `load_template.py` 加载具体大纲
- 浏览知识库节点 → 用 `search_graph_tree.py`

## 注意事项

- 大纲数据通过 `outline` 事件推送给前端，**绝对不要**在文字回复里输出大纲内容或任何 Markdown 格式的章节列表
- 每次脚本调用后，文字回复严格控制在 1-2 句话，只说状态结果和下一步询问
- 已有大纲时，用户的后续输入**优先理解为修改指令**，而不是新的分析请求
- 若用户只是聊天，正常回应，不调用任何脚本

## Agent 的视野边界

Agent 默认能看到**大纲**（system prompt 中的 YAML）。报告生成后，指标数据和总结文本也可按需查阅：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/get_report_data.py <node_id>
```

输出指定节点子树中所有 L5 指标的查询结果（每项最多 10 行）以及各层节点的总结文本。
报告尚未生成时输出提示。可用于：
- 了解某章节的实际数据，判断是否需要调整大纲
- 回答用户关于"某个指标结果是多少"的具体问题

**修改报告 = 修改大纲**，不存在直接编辑报告文本的途径：
- 用户说「把某一节的指标换掉」「删掉某个章节」→ 一律通过 `modify_outline.py` 修改大纲，再触发生成
- 前端增量更新是自动的：未变动的指标数据和子树总结会被复用，无需重新查询；只有大纲中新增或修改的部分才重新执行
- Agent 无需关心哪些内容需要重新生成，只需正确修改大纲并触发即可
