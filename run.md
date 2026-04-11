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