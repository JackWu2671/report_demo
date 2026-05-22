---
name: publish-knowledge
description: >
  知识发布工具包。将积累的专家模板批量分析并融合回知识图谱，扩展知识库覆盖范围。
  触发条件：用户主动说"发布知识"、"更新知识库"、"知识发布"，或点击了"发布知识"按钮。
  不适用：普通对话、报告生成、专家沉淀过程中的实时操作。
version: 1.0.0
author: report_demo
metadata:
  hermes:
    category: report
    tags: [knowledge, publish, graph, fusion, template]
---

# 知识发布

将专家沉淀的模板批量融合回知识图谱，让图谱随使用持续演进。
本 skill 独立于专家沉淀流程，由管理员或知识负责人定期主动触发。

所有工具均为 Python 脚本，通过 `bash` 调用。脚本路径：
`$SKILLS_DIR/publish-knowledge/scripts/<script>.py`

> **跨平台命令规则**：所有命令写在单行，JSON 参数用外层双引号 + 内层 `\"` 转义。

## 脚本工具参考

| 脚本 | 命令格式 | 说明 |
|------|---------|------|
| `list_templates.py` | `python3 ... [--with-outline]` | 列出所有已保存模板，`--with-outline` 附带完整大纲 |
| `show_graph.py` | `python3 ...` | 输出当前知识图谱完整结构（md_with_ids 格式） |
| `graph_manage.py` | `python3 ... --template-id <id> --add-nodes "[...]" --enrich-nodes "[...]"` | 将分析结果写入知识图谱 |

## 工作流程

### 步骤 1：了解当前状态

并行获取两份信息：

```bash
python3 $SKILLS_DIR/publish-knowledge/scripts/list_templates.py --with-outline
```

```bash
python3 $SKILLS_DIR/publish-knowledge/scripts/show_graph.py
```

`list_templates.py` 返回所有模板及其大纲；`show_graph.py` 返回当前知识图谱的完整节点树。

### 步骤 2：分析融合方案

对比模板大纲与知识图谱，识别模板中所有节点，逐一判断：

| 节点类型 | 判断 | 动作 |
|---------|------|------|
| L2/L3/L4 | 图谱中无语义相近节点，且有跨场景通用价值 | `add_node`：新增结构节点 |
| L2/L3/L4 | 图谱中已有语义相近节点，模板提供了新的业务视角 | `enrich_existing`：追加描述 |
| L2/L3/L4 | 过于场景专属，或与已有节点完全重复 | `ignore` |
| L5（query） | 图谱中不存在该 query 节点 | `add_node` level=5：新增 query 节点 |
| L5（query） | 图谱中已存在该 query 节点 | `ignore`（query 节点描述是查询参数，不追加） |

**L1 节点不参与融合**（顶层主题由人工维护）。

**add_node 必填字段：**
- `level`：2、3、4 或 5
- `name`：节点名称
- `keywords`：3～6 个检索关键词
- `description`：L2/L3/L4 为 50～100 字业务描述；**L5 的 description 即查询参数**，直接决定数据过滤范围，须准确描述查询意图（如"仅统计南宁市的企业行业分布"）
- `parent_id`：必须是知识图谱中已有节点的 ID（L5 的父节点为 L4）

**add_node 可选字段：**
- `condition`：展示条件，格式为"当……时，本节才展示"（如"当用户选择了城市维度时，本节才展示"）。仅在节点并非始终展示时填写，无条件限制则留空。

**enrich_existing 必填字段（仅适用于 L2/L3/L4）：**
- `node_id`：已有节点 ID（如 `L3_001`）
- `append`：30～80 字的补充描述，不重复已有内容

### 步骤 3：向用户呈现方案并确认

以清单形式列出分析结果，**等待用户确认后**再执行：

```
拟新增节点（X 个）：
  + [L3] 细颗粒功能板覆盖评估 → 父节点：L2_001（fgOTN部署）
    关键词：fgOTN, 覆盖评估, 站点选址
    描述：...
  ...

拟丰富描述（Y 个）：
  ~ L4_003（低阶交叉资源分析）
    追加：...
  ...

忽略（Z 个）：过于场景专属，暂不纳入图谱。

以上方案是否确认执行？
```

如用户要求调整，修改方案后再次确认。

### 步骤 4：执行融合

用户确认后，对每个来源模板分别调用（同一模板的所有操作合并为一次调用）：

```bash
python3 $SKILLS_DIR/publish-knowledge/scripts/graph_manage.py --template-id <template_id> --add-nodes "[{\"level\": 3, \"name\": \"节点名\", \"keywords\": [\"kw1\",\"kw2\"], \"description\": \"描述\", \"parent_id\": \"L2_001\", \"condition\": \"当用户选择了城市维度时，本节才展示\"}]" --enrich-nodes "[{\"node_id\": \"L3_001\", \"append\": \"补充描述\"}]"
```

`condition` 字段可省略（无展示条件的节点不传或传空字符串）。

没有任何变更的模板可跳过（`--add-nodes "[]" --enrich-nodes "[]"` 会正常执行但无写入）。

### 步骤 5：完成

汇报写入结果，**提醒重建 FAISS 索引**：

```
知识发布完成：新增 X 个节点，丰富描述 Y 个节点。
⚠️ 请运行 python scripts/build_index.py 重建向量索引，新节点才能被检索命中。
```

## 注意事项

- 融合是**累积操作**，已有节点的描述只追加不覆盖
- 每次发布前都要重新拉取最新图谱（`show_graph.py`），不要依赖上一次的结果
- 同一概念的节点只处理一次，多个模板描述同一概念时合并为一条 add_node 或 enrich_existing
