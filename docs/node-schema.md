# 大纲节点 Schema 设计

本文描述大纲树中节点的字段设计，包括结构节点与叶子节点的区分规则、各类型叶子节点的字段集，以及 description / summary 的静态与动态渲染逻辑。

---

## 一、核心规则：有无 children 决定节点角色

```
有 children  →  结构节点（Section）：在报告里输出一个章节标题
无 children  →  叶子节点（Leaf）  ：产出实际报告内容，由 type 决定内容来源
```

`type` 字段**只存在于叶子节点**，结构节点无需 type。

---

## 二、所有节点公共字段

| 字段 | 必须 | 说明 |
|------|------|------|
| `id` | ✅ | 节点唯一标识 |
| `name` | ✅ | 节点名称；结构节点作为章节标题输出到报告 |
| `description` | | 静态文本，直接渲染到报告开头；为空时看 `descriptionSuggestion` |
| `descriptionSuggestion` | | LLM prompt，看数据动态生成开头段落 |
| `summarySuggestion` | | LLM prompt，看数据动态生成结尾段落 |
| `summary` | | 静态文本，直接渲染到报告结尾；为空时看 `summarySuggestion` |
| `children` | | 子节点数组；有则为结构节点，无则为叶子节点 |

### description / summary 渲染逻辑

两者规则完全对称：

```
description 有内容                        → 静态，直接渲染
description 为空 + descriptionSuggestion 有值 → 报告执行时 LLM 动态生成
description 为空 + descriptionSuggestion 为空 → 不渲染

summary 有内容                            → 静态，直接渲染
summary 为空 + summarySuggestion 有值     → 报告执行时 LLM 动态生成
summary 为空 + summarySuggestion 为空     → 不渲染
```

`descriptionSuggestion` 生成的内容渲染在数据**之前**（开头），`summarySuggestion` 在数据**之后**（结尾）。两者均不回填回节点字段——数据每次可能变化，每次报告执行时重新生成以保持最新。

---

## 三、结构节点额外字段

结构节点的唯一职责是在报告中输出章节标题，并可附带条件控制。

| 字段 | 必须 | 说明 |
|------|------|------|
| `condition` | | 展示条件表达式；求值为假则跳过整个子树，不进报告 |
| `condition_queries` | | `condition` 依赖的叶子节点 name 列表；报告执行器必须先查这些节点才能求值条件 |

> `summarySuggestion` 挂在结构节点上时，汇总**所有子孙叶子节点**的数据；挂在叶子节点上时，只总结**自身**这一条数据。

---

## 四、叶子节点额外字段

### 公共（所有叶子节点）

| 字段 | 必须 | 说明 |
|------|------|------|
| `type` | ✅ | `"sql"` \| `"api"` \| `"narrative"` |

---

### type = "sql"

从数据库执行 SQL 查询，结果渲染为图表或表格。

| 字段 | 必须 | 说明 |
|------|------|------|
| `sql_config` | ✅ | SQL 数据源配置对象 |
| `sql_config.exec_sql` | ✅ | 执行的 SQL 语句 |
| `sql_config.tables` | ✅ | SQL 涉及的表名列表，作为取数 API 的路由参数 |
| `renderType` | | 渲染类型：`BAR` / `LINE` / `PIE` / `TABLE` |
| `colX` | | 图表 X 轴对应的结果列名 |
| `colY` | | 图表 Y 轴对应的结果列名 |

---

### type = "api"

从外部 API 获取数据，结果渲染为图表或表格。

| 字段 | 必须 | 说明 |
|------|------|------|
| `api_config` | ✅ | API 调用配置对象 |
| `api_config.api_name` | ✅ | API 名称 |
| `api_config.api_param` | | API 调用参数 |
| `renderType` | | 渲染类型：`BAR` / `LINE` / `PIE` / `TABLE` |
| `colX` | | 图表 X 轴对应的结果列名 |
| `colY` | | 图表 Y 轴对应的结果列名 |

> `renderType` / `colX` / `colY` 是渲染配置，与数据来源（sql/api）正交——不管数据怎么取，渲染逻辑相同。

---

### type = "narrative"

静态 Markdown 内容，直接渲染到报告，不触发任何数据查询。适用于排查路径、操作步骤、代码块、流程说明等自由内容。

| 字段 | 必须 | 说明 |
|------|------|------|
| `content` | ✅ | Markdown 文本，直接渲染到报告 |

---

## 五、报告中的节点执行与渲染顺序

执行顺序和渲染（输出）顺序不同——`descriptionSuggestion` 需要看子节点数据，所以必须等数据收集完才能生成，但最终在报告里仍然出现在数据之前。

**执行顺序（谁先跑）：**

```
① 收集所有子节点 / 叶子节点数据（SQL 查询、API 调用）
  ↓
② LLM 看数据生成 description（若有 descriptionSuggestion）
  ↓
③ LLM 看数据生成 summary（若有 summarySuggestion）
  ↓
④ 按渲染顺序输出报告
```

**渲染顺序（报告里的呈现位置）：**

```
章节标题（name）
  ↓
description（静态）/ descriptionSuggestion 生成的开头段落
  ↓
子节点内容（递归渲染）/ 叶子节点数据（图表、表格、narrative 文本）
  ↓
summary（静态）/ summarySuggestion 生成的结尾段落
```

---

## 六、完整节点示例

### 结构节点

```json
{
  "id": "L3_014",
  "name": "传送网络覆盖企业分析",
  "description": "本节评估现网 OTN 站点对高价值企业的覆盖情况。",
  "descriptionSuggestion": "",
  "summarySuggestion": "按价值层级总结覆盖缺口，给出部署优先级建议。",
  "summary": "",
  "condition": "",
  "condition_queries": [],
  "children": []
}
```

### sql 叶子节点

```json
{
  "id": "L5_318",
  "name": "高价值企业 OTN 覆盖率",
  "type": "sql",
  "sql_config": {
    "exec_sql": "SELECT 行政区, 覆盖率 FROM dwd_otn_site ...",
    "tables": ["dwd_otn_site"]
  },
  "renderType": "BAR",
  "colX": "行政区",
  "colY": "覆盖率",
  "summarySuggestion": "指出覆盖率最低的三个区域及差距。",
  "summary": ""
}
```

### api 叶子节点

```json
{
  "id": "new_L5_001",
  "name": "近 7 日告警趋势",
  "type": "api",
  "api_config": {
    "api_name": "alert_trend",
    "api_param": {
      "days": 7,
      "level": "critical"
    }
  },
  "renderType": "LINE",
  "colX": "日期",
  "colY": "告警数",
  "summarySuggestion": "分析告警高峰时段，说明是否有收敛趋势。",
  "summary": ""
}
```

### narrative 叶子节点

```json
{
  "id": "new_L4_001",
  "name": "排查步骤",
  "type": "narrative",
  "content": "## 排查步骤\n\n1. 检查连接池配置\n```bash\nkubectl describe cm db-config\n```\n2. 查看慢查询日志..."
}
```
