# SPDX-License-Identifier: Apache-2.0
skill_executor_agent = LlmAgent(
    name="skill_executor_agent",
    model="gemini-flash-latest",
    instruction="""
You are a skill execution agent.

You are given:
- the selected skill
- the skill execution guide
- the current execution state
- the available tools

Decide the next step:
- which tool to call next
- with what arguments
- or whether to stop and answer

Only follow the selected skill.
""",
    tools=[
        search_nodes,
        get_edges_for_node,
        search_edges,
        get_node_by_id,
        search_around_node,
    ],
)

@node(name="run_skill")
async def run_skill(ctx, user_query: str, selected_skill: dict):
    state = {
        "user_query": user_query,
        "skill_name": selected_skill["name"],
        "skill_execution_guide": selected_skill["execution_text"],
        "history": [],
        "done": False,
    }

    max_steps = 8

    for _ in range(max_steps):
        decision = await ctx.run_node(skill_executor_agent, state)

        # 这里假设 decision 是结构化输出
        if decision["action"] == "finish":
            return decision["final_answer"]

        elif decision["action"] == "tool_call":
            tool_name = decision["tool_name"]
            tool_args = decision["tool_args"]

            # 简单 dispatch
            if tool_name == "search_nodes":
                tool_result = await ctx.run_node(search_nodes, tool_args)
            elif tool_name == "get_edges_for_node":
                tool_result = await ctx.run_node(get_edges_for_node, tool_args)
            elif tool_name == "search_edges":
                tool_result = await ctx.run_node(search_edges, tool_args)
            elif tool_name == "get_node_by_id":
                tool_result = await ctx.run_node(get_node_by_id, tool_args)
            elif tool_name == "search_around_node":
                tool_result = await ctx.run_node(search_around_node, tool_args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            state["history"].append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
            })

        else:
            raise ValueError(f"Unknown action: {decision['action']}")

    return "Skill execution stopped after max steps without a final answer."

from google.adk import Workflow

@node(name="skill_runtime_workflow", rerun_on_resume=True)
async def skill_runtime_workflow(ctx, user_query: str):
    skill_catalog = await ctx.run_node(load_skill_catalog)
    router_result = await ctx.run_node(select_skill, user_query, skill_catalog)

    selected_skill_name = router_result["selected_skill_name"]
    selected_skill = await ctx.run_node(
        load_selected_skill,
        selected_skill_name,
        skill_catalog,
    )

    final_answer = await ctx.run_node(
        run_skill,
        user_query,
        selected_skill,
    )
    return final_answer

root_agent = Workflow(
    name="skill_native_runtime",
    edges=[("START", skill_runtime_workflow)],
)