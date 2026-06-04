# 大纲与报告的数据设计

系统对"大纲"和"报告"各维护多种并行的数据形式，每种形式服务于不同的消费者。
本文解释每种形式为什么必须存在、不能合并。

---

## 一、大纲三份数据

大纲在前端 `ChatView.jsx` 中以三个平行的 state 存储，由后端推送 `outline` SSE 事件时同时更新。

| 数据 | State 变量 | SSE 字段 | 消费者 |
|------|-----------|---------|--------|
| 用户可读 Markdown | `outlineMd` | `evt.markdown` | 用户阅读 |
| YAML（精简视图） | `outlineLlm` | `evt.outline_yaml` | 大模型上下文 |
| JSON 树 | `outlineJson` + `outlineJsonRef` | `evt.outline_tree` | 前端报告生成逻辑 |

### 为什么三份都不能省

**能不能只存 JSON，让 LLM 直接读 JSON？**

不行。同一份大纲，两种格式的 token 量差异巨大：

```
# outline_yaml（精简 YAML）：~300 tokens
- id: L3_001
  name: 50GPON价值站点分析
  children:
    - id: L4_018
      name: 50GPON升级站点-套餐和超标
      condition: ${number("AEC覆盖用户数")>0}
      condition_queries: [AEC覆盖用户数]
      children:
        - id: L5_001
          name: AEC覆盖用户数

# JSON（完整树）：~2000+ tokens（含 summarySuggestion、renderType、exec_sql 等字段）
{ "id": "L3_001", "name": "50GPON价值站点分析", "level": 3,
  "description": "...", "condition": "", "condition_queries": [],
  "summarySuggestion": "", "renderType": "", "exec_sql": "...", "children": [ ... ] }
```

`outline_yaml` 是专为 LLM 设计的精简视图：只保留 LLM 需要随时引用的字段（id/name/description/condition/condition_queries），`level`、SQL 字段（exec_sql/apiName 等）、summarySuggestion 等字段默认省略——LLM 需要时可调用 `get_node_detail.py <node_id>` 按需拉取，而不是污染每次请求的上下文。

**能不能只存 outline_yaml，不要 JSON？**

不行。`outlineJson` 是程序逻辑必须的结构化数据：`buildSkeleton()` 遍历树生成报告骨架、`findNodeById()` / `findNodeByName()` 查节点、`replayCaches()` 校验 metric / summary 缓存是否仍然有效（metric 缓存按 `exec_sql` 等字段组成的签名失效，summary 缓存按整节点 JSON 失效）——这些都依赖字段访问和递归遍历，文本格式无法满足。

**能不能只存 JSON 和 outline_yaml，不要用户可读版？**

不行。用户可读版由 `MarkdownOutline.jsx` 渲染，展示层级颜色标签、描述文字、条件提示；outline_yaml 的原始文本格式不适合直接给用户看。

三份数据职责完全不重叠，每份都有唯一消费者，都不能省。

---

### 各份数据详解

#### 用户可读 Markdown — `outlineMd`

纯净的 Markdown 文本，不含节点 ID：

```
# 50GPON价值站点分析
## 50GPON升级站点-套餐和超标
### AEC覆盖用户数
```

由 `MarkdownOutline.jsx` 渲染，展示带颜色层级标签（L1–L5）、描述文字和条件提示。

#### YAML 精简视图 — `outlineLlm`

注入到 Agent 的 system prompt（`build_messages` 中 `## 当前大纲` 部分），让大模型在多轮对话中能通过 ID 修改特定节点——改属性值用 `edit_node` 工具（如 `exec_sql`、`name`），调结构（增删节点）用 `modify_outline.py`。格式省略 level、SQL 字段和空字段，只保留 LLM 需要随时引用的内容：

```yaml
- id: L3_001
  name: 50GPON价值站点分析
  description: 针对50GPON站点的套餐价值和升级潜力进行评估
  children:
    - id: L4_018
      name: 50GPON升级站点-套餐和超标
      condition: ${number("AEC覆盖用户数")>0}
      condition_queries:
        - AEC覆盖用户数
      children:
        - id: L5_001
          name: AEC覆盖用户数
```

LLM 如需查看某节点的完整信息（summarySuggestion、exec_sql、renderType 等），可调用：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/get_node_detail.py L4_018
```

#### JSON 树 — `outlineJson` / `outlineJsonRef`

完整的结构化树对象，是大纲**唯一权威数据源**，markdown / yaml 两个视图都从它派生。它保留全部字段，供程序逻辑（报告骨架生成、缓存校验、SQL 执行）使用。

一个 **L5 叶子节点**（携带 SQL 的查询指标），字段全集：

```json
{
  "id": "L5_318",
  "name": "高价值2B企业区域OTN站点覆盖率",
  "level": 5,
  "description": "",
  "condition": "",
  "condition_queries": [],
  "summarySuggestion": "",
  "renderType": "BAR",
  "colX": "行政区",
  "colY": "覆盖率",
  "apiName": "NL2SQL",
  "exec_sql": "select `行政区`,`覆盖率` from dwd_otn_site ...",
  "extracted_table": ["dwd_otn_site"],
  "children": []
}
```

一个 **L1–L4 结构节点**（章节，无 SQL）：SQL 相关字段全为空，靠 `children` 往下挂。

```json
{
  "id": "L3_014",
  "name": "传送网络覆盖企业分析",
  "level": 3,
  "description": "本节评估现网OTN站点对高价值2B企业的覆盖情况……",
  "condition": "",
  "condition_queries": [],
  "summarySuggestion": "按价值层级总结覆盖缺口，给出部署优先级建议",
  "renderType": "", "colX": "", "colY": "", "apiName": "",
  "exec_sql": "", "extracted_table": [],
  "children": [ /* L4 节点…… */ ]
}
```

##### 逐字段说明

| 字段 | 类型 | 主要层级 | 含义 |
|------|------|---------|------|
| `id` | string | 全部 | 节点唯一标识；**前缀编码层级**（`L3_014`→L3，`new_L2_001`→L2）。L5 的 id 必须引用知识库已有 query 节点，不可新建 |
| `name` | string | 全部 | 节点名。**L1–L4 是章节标题；L5 的 `name` 就是查询语句本身**——它是指标的唯一标识，系统据此定位/执行 SQL |
| `level` | int | 全部 | 1–5，依次为 场景 / 子场景 / 评估维度 / 评估项 / 评估指标。**`level==5` 是"查询叶子"的判定标志**，报告执行器据此把它当 SQL 指标处理；L1–L4 是纯结构节点 |
| `description` | string | L1–L4 | 50–100 字章节说明，进报告正文、也帮 LLM 理解。**L5 永远为空**（`name` 已是查询语句，写描述无意义且被禁止） |
| `condition` | string | 任意 | 展示条件表达式，如 `${number("AEC覆盖用户数")>0}`。求值为假则该节点从报告中跳过（后端推 `report_skip` 事件） |
| `condition_queries` | string[] | 任意 | `condition` 依赖的指标名列表。报告执行器必须**先查这些指标**，才能求值 `condition` 决定是否展示本节点 |
| `summarySuggestion` | string | L1–L4（章节） | 生成该章节小结时给 LLM 的提示/侧重点 |
| `renderType` | string | **仅 L5** | 可视化类型：`BAR` / `LINE` / `PIE` / `TABLE`。报告执行器据此决定推图表事件还是表格事件（详见 `report-generation.md`） |
| `colX` | string | **仅 L5** | 图表 X 轴（柱/线的类目轴，饼图的名称键）对应的**结果列名** |
| `colY` | string | **仅 L5** | 图表 Y 轴（数值轴，饼图的数值键）对应的**结果列名** |
| `apiName` | string | **仅 L5** | 该指标 SQL 的来源 API（如 `NL2SQL`），属溯源元数据；运行时取数只用 `exec_sql` + `extracted_table`，不直接用它 |
| `exec_sql` | string | **仅 L5** | 该指标实际执行的 SQL 语句。常含反引号包裹的中文列名，故改它走 `edit_node` 工具而非 bash |
| `extracted_table` | string[] | **仅 L5** | 该 SQL 涉及的表名，作为取数 API 的 `table` 参数传入 |
| `children` | node[] | 全部 | 子节点数组。**L5 必为空**（叶子），L1–L4 至少挂一个子节点 |

##### 字段分两类：结构节点 vs 查询叶子

把字段按"谁拥有"分开看，就理解了为什么 `to_yaml`（LLM 视图）能大胆省略大半字段：

- **L1–L4 结构节点**只用 `id` / `name` / `level` / `description` / `summarySuggestion` / `condition` / `children`——SQL 那一组字段对它们恒为空。
- **L5 查询叶子**才携带 SQL 执行所需的一组：`renderType` / `colX` / `colY` / `apiName` / `exec_sql` / `extracted_table`。

`to_yaml` 省掉 `level`（id 前缀已隐含）和整组 SQL 字段：对 L1–L4 它们本就是空，对 L5 则 LLM 平时不需要——需要时用 `get_node_detail.py <node_id>` 按需拉取，避免每轮上下文被 SQL 字符串撑大。

---

##### `outlineJsonRef`：为什么还要一个 ref

`outlineJsonRef` 是与 `outlineJson` 同步的 `useRef`，解决 React setState 异步问题。

**背景**：`setState` 不会立刻改变变量，新值要等到下一次 render 才可读。后端有时在同一批 SSE 事件里连续推送 `outline` 和 `start_report`，两个事件在同一次 JavaScript 执行里依次处理，React 来不及 re-render：

```js
// 事件1：收到新大纲
outlineJsonRef.current = evt.outline_tree   // 立刻生效
setOutlineJson(evt.outline_tree)            // 安排更新，但还没 re-render

// 事件2：立刻触发生成报告（React 还没 re-render，outlineJson 仍是旧值）
generateReport()
  → const tree = outlineJsonRef.current   // ✅ 拿到新树
  → const tree = outlineJson              // ❌ 仍是 null（旧值），报告不会生成
```

---

## 二、大纲的四种后端表示

在后端，大纲通过 `outline_utils.py` 在四种表示之间转换：

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `to_markdown(tree)` | outline_tree | 纯 Markdown | 前端用户视图 |
| `to_yaml(tree)` | outline_tree | YAML 精简文本 | LLM 上下文（system prompt，只读） |
| `to_clean_json(tree)` | outline_tree | 干净 JSON dict | 前端 JSON 树、存储 |
| `from_data(obj)` | JSON 节点结构（list/dict） | outline_tree | `set_outline` 工具：LLM 传 JSON 数组 → 建回树 |

唯一权威数据源是 `outline_tree`（dict）。`to_yaml()` 省略 level 和 SQL 字段；`from_data()` 通过 ID 前缀（`L1_xxx` → level 1，`new_L2_xxx` → level 2）自动推断 level。

> **历史说明**：早期 `set_outline` 收 YAML 文本、用 `from_yaml()` 解析。但 YAML 的结构靠换行/缩进承载，LLM 经工具参数传长 YAML 时常被压成一行导致解析失败，故改为收 JSON 节点数组、走 `from_data()`——JSON 结构靠括号承载，不依赖空白格式。`from_yaml` 已移除。

---

## 三、报告两份数据

报告在前端对应两个 Tab，数据来源于后端 `/api/report` SSE 流。

| 数据 | State 变量 | 消费者 |
|------|-----------|--------|
| 渲染报告 | `report` + `chartData` + `tableData` | 用户阅读 |
| Markdown 源码 | `report`（同上） | 调试 / 复制 |

两个 Tab 共享同一个 `report` state，区别只是渲染方式——"报告"Tab 通过 `ReportView.jsx` 渲染为带图表的页面，"Markdown"Tab 用 `<pre>` 展示原始文本。

**骨架说明**：`buildSkeleton(tree)` 生成的带占位符模板（`<span data-ph="...">` / `<span data-ph-summary="...">`）作为 `generateReport()` 内部局部变量，不单独存储。`report` state 以骨架为初始值，随 SSE 事件持续替换占位符直到报告完成。

---

## 四、数据流总览

```
后端 SSE 'outline' 事件
    ├─ evt.markdown      ──→  outlineMd   ──→  MarkdownOutline（用户阅读）
    ├─ evt.outline_yaml  ──→  outlineLlm  ──→  Agent system prompt（LLM 修改大纲）
    └─ evt.outline_tree  ──→  outlineJson ──→  buildSkeleton() / findNodeById() 等逻辑
                                │
                                ▼
                          buildSkeleton()  （局部变量 sk）
                                │
                          replayCaches()   （复用上一次缓存）
                                │
                                ▼
后端 SSE /api/report 流
    ├─ report_metric  ──→  替换 data-ph 占位符   ─┐
    ├─ report_summary ──→  替换 data-ph-summary  ─┤─→  report ──→ ReportView（渲染）
    └─ report_done    ──→  结束标记               ─┘         └──→ <pre>（Markdown 源码）
```
