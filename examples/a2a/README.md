# A2A Local Flow Examples

This folder mirrors the local Agent2Agent (A2A) development flow:

1. Define `AgentCard`
2. Define ADK `LlmAgent`
3. Define `AgentExecutor` (wrapping ADK `Runner`)
4. Create local `A2aAgent`
5. Call test endpoints (`handle_authenticated_agent_card`, `on_message_send`, `on_get_task`)

## Files

- `01_agent_card.py`
- `02_llm_agent.py`
- `03_agent_executor.py`
- `04_local_agent.py`
- `05_test_local_calls.py`

## Notes

- These are reference examples for call flow.
- They follow Google Vertex AI Agent Builder A2A docs:
  - <https://docs.cloud.google.com/agent-builder/agent-engine/develop/a2a>
