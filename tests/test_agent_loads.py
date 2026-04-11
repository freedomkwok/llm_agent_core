from agent import root_agent


def test_root_agent_exists() -> None:
    assert root_agent is not None
    assert root_agent.name == "imp_chat_agent"
