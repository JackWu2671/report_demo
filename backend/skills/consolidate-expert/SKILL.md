---
name: consolidate-expert
description: >
  专家知识沉淀工具包。用户发来一段较长的业务描述（通常 80～300 字），
  内容是他自己的分析判断、工作方法或场景经验，而不是在提问——这就是触发信号。
  典型表现：用陈述句描述"我们一般怎么看……"、"这个场景需要关注……"、
  "根据我的经验……"、"做这类分析要先……"，或者直接把一段业务思路一次性发过来。
  无论用户有没有说"保存"或"沉淀"，只要是在输出自己的业务知识，就必须加载此 skill。
  不适用于：用疑问句提问、请求生成报告、与业务知识分享无关的简短对话。
version: 3.0.0
author: report_demo
metadata:
  hermes:
    category: report
    tags: [expert, knowledge, template, consolidation]
---

# 沉淀专家知识

专家的业务经验分散在脑子里，无法被系统复用。这个 skill 的目标是把专家的经验提炼成
结构化模板，让后续所有报告都能站在专家的肩膀上。

所有工具均为 Python 脚本，通过 `bash` 调用。脚本路径：
`$SKILLS_DIR/consolidate-expert/scripts/<script>.py`

> **跨平台命令规则**：所有命令必须写在**单行**，不得使用 `\` 换行续接（Windows 不支持）。

## 脚本工具参考

| 脚本 | 说明 |
|------|------|
| `search_graph_tree.py "查询词"` | 检索知识图谱节点（与 analyze-network 共用脚本） |
| `get_node_detail.py <node_id> [...]` | 查询节点完整信息，与 analyze-network 共用脚本 |
| `set_outline.py`（从 stdin 读取） | 解析 YAML 大纲写入会话，推送给前端 |
| `set_metadata.py --scene-name "..." --summary "..." --keywords "kw1,kw2" --usage-conditions "..."` | 写入场景元数据 |
| `save_template.py` | 将当前大纲和元数据保存为模板 |

**注意**：`search_graph_tree.py` 位于 analyze-network 脚本目录：
`$SKILLS_DIR/analyze-network/scripts/search_graph_tree.py`

## 工作流程

### 步骤 1：检索知识库

```bash
python3 $SKILLS_DIR/analyze-network/scripts/search_graph_tree.py "专家描述原文"
```

返回带祖先路径的节点树，包含可用的 query 节点 id。记住这些 id，后续构造大纲时 query 节点必须引用知识库已有 id，不可新建。

- `success` → 进入步骤 2
- `not_found` → 告知专家当前知识库暂不覆盖该场景

### 步骤 2：构造大纲

根据专家输入和知识库节点，自行设计大纲结构，以 YAML 格式通过 stdin 传入：

```bash
python3 $SKILLS_DIR/consolidate-expert/scripts/set_outline.py << 'EOF'
- id: new_001
  name: 标题
  description: 50～100字描述
  children:
    - id: new_002
      name: 章节
      description: 描述
      children:
        - id: new_003
          name: 节
          description: 描述
          children:
            - id: new_004
              name: 小节
              description: 描述
              children:
                - id: L5_001
                  name: query节点名称
EOF
```

YAML 大纲约束（违反任意一条视为无效输出）：
1. 顶层必须恰好一个节点（即一个 L1），作为大纲根节点（报告总标题）
2. L5 query 节点必须是叶子节点，禁止在其下挂任何子节点
3. 禁止新建 query 节点（即禁止 `id: new_xxx` 且无子节点的叶子节点），L5 只能引用 search_graph_tree 返回的知识库节点 id
4. 禁止 L4 新建节点（`id: new_xxx`）作叶子节点，每个新建 L4 下方必须至少挂一个知识库已有的 query 节点
5. 所有新建节点（`id: new_xxx`）必须填写 `description`（50～100 字）
6. L2/L3/L4 由你按专家意图自由设计，不得用知识库节点名称替代专家描述的分析板块名称
7. `condition`/`condition_queries` 按需填写，其他字段（level、exec_sql 等）不要写入 YAML

调用后大纲立即展示给专家。

### 步骤 3：填写场景元数据

`set_outline.py` 调用完毕后，**立即**调用：

```bash
python3 $SKILLS_DIR/consolidate-expert/scripts/set_metadata.py --scene-name "传送网络覆盖分析" --summary "面向OTN站点企业覆盖现状的专项分析，识别覆盖缺口与部署机会" --keywords "OTN,企业覆盖,fgOTN,站点部署,覆盖缺口" --usage-conditions "适用于需要评估OTN网络企业覆盖现状、识别部署优先级的场景"
```

字段要求：
- `--scene-name`：中文，不超过 10 字
- `--summary`：一句话摘要，不超过 50 字
- `--keywords`：逗号分隔，3～8 个核心领域关键词
- `--usage-conditions`：适用条件，不超过 80 字

### 步骤 4：按专家意见修改（按需）

使用 analyze-network 的 `modify_outline.py` 修改大纲（外层双引号，内层 `\"` 转义）：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/modify_outline.py "[{\"op\": \"modify_node_name\", \"node_id\": \"new_002\", \"value\": \"新名称\"}]"
```

修改后一句话确认变更，询问是否满意。

### 步骤 5：保存为模板

只在专家明确确认时调用（说"保存"、"就这样"、"好的"等），不要主动催促：

```bash
python3 $SKILLS_DIR/consolidate-expert/scripts/save_template.py
```

成功时输出 `{"template_id": "...", "scene_name": "...", "path": "..."}`，告知专家模板名称和存储路径，流程结束。

> 知识图谱融合由独立的 `publish-knowledge` skill 批量处理，不在本流程内触发。

## 注意事项

- 大纲通过独立事件推送给前端，**绝对不要**在文字回复里输出大纲内容或任何 Markdown 格式的结构数据
- 每次脚本调用后，文字回复严格控制在 1-2 句话，只说结论和下一步询问
