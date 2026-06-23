# 大纲节点 Schema 设计

本文描述大纲树中节点的字段设计与渲染逻辑。

---

## 一、设计原则

所有节点统一结构，没有显式 type 字段。节点有哪些字段就渲染哪些内容——`sql_config` / `api_config` / `content` 可以单独存在，也可以同时存在于一个节点上。

`children` 字段独立存在，任何节点都可以有子节点。

---

## 二、节点字段全集

| 字段 | 必须 | 说明 |
|------|------|------|
| `id` | ✅ | 节点唯一标识 |
| `name` | ✅ | 节点名称，渲染为章节标题 |
| `description` | | 静态文本，渲染在数据之前（开篇） |
| `descriptionSuggestion` | | prompt，LLM 看数据后生成开篇段落 |
| `content` | | 静态 Markdown 正文，直接渲染 |
| `sql_config` | | SQL 数据源配置，有则执行查询 |
| `api_config` | | API 数据源配置，有则调用接口 |
| `renderType` | | 渲染类型：`BAR` / `LINE` / `PIE` / `TABLE`（sql / api 共用） |
| `colX` | | 图表 X 轴对应的结果列名 |
| `colY` | | 图表 Y 轴对应的结果列名 |
| `summarySuggestion` | | prompt，LLM 看数据后生成收尾段落 |
| `summary` | | 静态文本，渲染在数据之后（收尾） |
| `condition` | | 展示条件表达式，求值为假则跳过该节点及其子树 |
| `condition_queries` | | `condition` 依赖的数据节点名列表，需提前查询 |
| `children` | | 子节点数组 |

### sql_config 字段

| 字段 | 必须 | 说明 |
|------|------|------|
| `sql_config.exec_sql` | ✅ | 执行的 SQL 语句 |
| `sql_config.tables` | ✅ | SQL 涉及的表名列表，作为取数 API 的路由参数 |

### api_config 字段

| 字段 | 必须 | 说明 |
|------|------|------|
| `api_config.api_name` | ✅ | API 名称 |
| `api_config.api_param` | | API 调用参数 |

---

## 三、执行顺序与渲染顺序

两者分离：执行时数据先行，LLM 生成在后；渲染时位置固定不变。

**执行顺序（谁先跑）：**

```
① 收集本节点数据（sql_config 查询 / api_config 调用）
② 递归处理 children，收集子节点数据
③ LLM 看全部数据生成 descriptionSuggestion 内容（若有）
④ LLM 看全部数据生成 summarySuggestion 内容（若有）
⑤ 按渲染顺序输出
```

**渲染顺序（报告中的位置，固定不变）：**

```
heading（name）
description（静态）/ descriptionSuggestion 生成内容   ← 永远在最前
sql_config 查询结果
api_config 调用结果
content（静态 Markdown）
children（递归渲染）
summary（静态）/ summarySuggestion 生成内容           ← 永远在最后
```

`descriptionSuggestion` 和 `summarySuggestion` 都是 LLM 看完数据后生成，区别只在渲染位置：description 是开篇分析，summary 是收尾结论。

### description / summary 渲染规则

```
有静态文本（description / summary 不为空）→ 直接渲染
无静态文本 + 有 Suggestion                → LLM 动态生成后渲染到对应位置
两者均为空                                → 该位置不渲染
```

---

## 四、节点示例

### 纯结构节点（只有标题和子节点）

```json
{
  "id": "L3_014",
  "name": "传送网络覆盖分析",
  "description": "本节评估现网 OTN 站点对高价值企业的覆盖情况。",
  "summarySuggestion": "按价值层级总结覆盖缺口，给出部署优先级建议。",
  "condition": "",
  "condition_queries": [],
  "children": []
}
```

### SQL 查询节点

```json
{
  "id": "L5_318",
  "name": "高价值企业 OTN 覆盖率",
  "sql_config": {
    "exec_sql": "SELECT 行政区, 覆盖率 FROM dwd_otn_site ...",
    "tables": ["dwd_otn_site"]
  },
  "renderType": "BAR",
  "colX": "行政区",
  "colY": "覆盖率",
  "summarySuggestion": "指出覆盖率最低的三个区域及差距。"
}
```

### API 查询节点

```json
{
  "id": "new_L5_001",
  "name": "近 7 日告警趋势",
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
  "summarySuggestion": "分析告警高峰时段，说明是否有收敛趋势。"
}
```

### 静态内容节点

```json
{
  "id": "new_L4_001",
  "name": "排查步骤",
  "content": "## 排查步骤\n\n1. 检查连接池配置\n```bash\nkubectl describe cm db-config\n```\n2. 查看慢查询日志..."
}
```

### 混合节点（SQL 数据 + 静态说明同时存在）

```json
{
  "id": "new_L3_001",
  "name": "异常站点分析",
  "descriptionSuggestion": "根据查询结果，概括本节异常站点的整体情况。",
  "sql_config": {
    "exec_sql": "SELECT 站点, 异常类型, 数量 FROM dwd_alarm ...",
    "tables": ["dwd_alarm"]
  },
  "renderType": "TABLE",
  "content": "### 处理建议\n\n- 优先处理持续超过 24 小时的告警\n- 联系属地工程师确认现场情况",
  "summarySuggestion": "总结各类异常的分布规律，给出优先级排序建议。",
  "children": []
}
```
