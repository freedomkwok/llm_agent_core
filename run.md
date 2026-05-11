<!-- SPDX-License-Identifier: Apache-2.0 -->

## Setup and run

```bash
make setup
cp .env.example .env
make run
```

## ADK web mode

```bash
make web
```

## Core architecture notes

工业系统通常把 `planner / router / executor` 分开：

1. 用户输入
2. Router 选择 skill（prompt 要短）
3. Executor 基于该 skill 执行
4. 如有 tool calls，执行后回传结果
5. 重复直到产出最终答案

1. Skill Registry：定义地图
2. Agent Registry：定义执行者
3. Planner：把地图展开成执行计划
4. Runtime：真的去跑
对的
Phase 3

Planner + workflow expansion
	•	skill 变成 map root
	•	自动展开 subskill DAG / graph

这个是我要做的
 graph-driven iterative planner
 incremental planning
或者
graph-guided step expansion


 RRF is fusion, MMR is diversification, 
 if you want something cheaper/faster and more forgiving. RRF is a strong baseline because it blends keyword and semantic matching well. For skill routing, that matters when your skill names, labels, triggers, and descriptions may match partly by exact phrasing and partly by semantic similarity.  

 cross encoder is semantic pairwise scoring, 
 because routing is usually a precision problem, not a diversity problem. You usually want “which single skill best matches the current task?” not “give me a broad, non-redundant spread of skills.” Zep explicitly positions cross encoder as the best choice when accuracy in relevance scoring matters most. That lines up very well with routing

 episode mentions is recurrence-based,
and node distance is graph-topology-based

1.	Search skills with scope="nodes" using a strong skill query.
	2.	Start with reranker="cross_encoder".
	3.	Take top 5–10 candidate skills.
	4.	Then do your own app-level rerank using fields like applies_when, not_applies_when, dependency readiness, and risk penalties.


