# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from agents.sync_langfuse_prompts import agent_prompt_specs, sync_prompts, zep_prompt_specs


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.prompts: list[dict] = []
        self.flushed = False

    def create_prompt(self, **kwargs):
        self.prompts.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def test_zep_prompt_specs_include_main_and_sub_agent_prompts() -> None:
    specs = zep_prompt_specs()

    assert [spec.name for spec in specs] == [
        "agents/zep_query_agent/instruction",
        "agents/zep_query_agent/instruction_subagent",
    ]
    assert specs[0].labels == ["production"]
    assert specs[1].labels == ["production"]
    assert "Zep Agent" in specs[0].prompt
    assert "create subagent" in specs[1].prompt
    assert agent_prompt_specs(agent="zep_agent") == specs


def test_sync_prompts_creates_text_prompts_and_flushes() -> None:
    fake_client = FakeLangfuseClient()
    specs = zep_prompt_specs()

    synced = sync_prompts(specs, client=fake_client)

    assert synced == [
        "agents/zep_query_agent/instruction",
        "agents/zep_query_agent/instruction_subagent",
    ]
    assert [prompt["name"] for prompt in fake_client.prompts] == synced
    assert all(prompt["type"] == "text" for prompt in fake_client.prompts)
    assert fake_client.prompts[1]["prompt"] == (
        "create subagent when task should be split into a new agent for search"
    )
    assert fake_client.flushed is True
