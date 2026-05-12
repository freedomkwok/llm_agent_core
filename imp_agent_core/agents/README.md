<!-- SPDX-License-Identifier: Apache-2.0 -->

## Agents Flow

This folder contains agent building blocks and shared execution infrastructure.

### Current pieces

- `agent_core/`
  - Shared executor/orchestration code.
  - `adk/executor.py` holds the reusable A2A + ADK Runner flow:
    - create/get ADK session
    - use in-memory session/memory/artifact services
    - run the ADK agent
    - wrap execution with Langfuse chain tracing
    - return result back through A2A task updates

- `planning_agent/`
  - Current planner-style agent.
  - `planning_agent.py` contains the planning engine logic and schema-based planning generation.
  - `planning_adk_agent.py` is the custom ADK `BaseAgent` that calls the planning engine.
  - `a2a_executor.py` is now only a thin wrapper over `agent_core`.
  - `start_agent.py` builds the local A2A agent instance.

### Current planning flow

1. Host sends a request to the A2A agent.
2. `PlanningA2aExecutor` receives the request.
3. Shared logic in `agent_core/adk/executor.py`:
   - starts task updates
   - creates/loads the ADK session
   - runs the ADK custom agent through `Runner`
   - tracks the chain in Langfuse
4. `PlanningInferenceAdkAgent` calls `PlanningInferenceEngine`.
5. `PlanningInferenceEngine` produces a `PlanningSchema`.
6. The plan is rendered to text and returned as the task artifact.
7. The A2A task is marked complete.

Important: the current planning agent does **not** execute the plan. It only returns planning output.

### Two orchestration modes

#### 1. Host-driven orchestration

The host/backend coordinates multiple agents.

Example:

1. Host calls planner/router.
2. Planner returns plan or selected skill.
3. Host calls executor.
4. Host may repeat until done.

Use this when:

- you want simple, explicit control in backend code
- different services own different execution stages
- extra request/response hops are acceptable

#### 2. Agent-internal orchestration

One top-level brain/orchestrator agent coordinates internally.

Example:

1. Host calls one top-level agent.
2. That agent queries routing sources such as `zep_graph`.
3. It selects the right skill or sub-agent.
4. It executes internally and loops until done.
5. It returns only the final result to the host.

Use this when:

- you want fewer host roundtrips
- planner/router/executor should feel like one subsystem
- the "brain" should own the loop and stop conditions

### Intended future direction

If the planning agent becomes the real "brain", it will likely evolve into:

- request understanding
- `zep_graph` skill lookup
- skill selection/routing
- skill loading
- execution loop
- final completion signal

At that point, planning will be only one stage inside a broader orchestrator flow.

### A2A note

When using A2A as the boundary:

- `on_message_send(...)` starts the task
- `on_get_task(...)` fetches task status/result

These are transport/API calls, not the orchestration logic itself.

### Sync agent prompts to Langfuse

Agent prompt names live in each agent's `config.yaml`. To create or update
those prompts in the configured Langfuse project:

```bash
uv run python -m agents.sync_langfuse_prompts --agent zep_agent --dry-run
uv run python -m agents.sync_langfuse_prompts --agent zep_agent
```

The command loads repo `.env` through `agents.zep_agent._env`, including
`MAP_LANGFUSE_*` to `LANGFUSE_*` mapping.
