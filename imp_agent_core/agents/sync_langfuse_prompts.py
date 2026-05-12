# SPDX-License-Identifier: Apache-2.0
"""Sync agent prompt definitions from agent configs into Langfuse."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from imp_agent_core.agents.zep_agent._env import bootstrap_env
from langfuse import get_client

_AGENTS_DIR = Path(__file__).resolve().parent
_DEFAULT_AGENT = "zep_agent"
_DEFAULT_TAGS = ["imp_agent_core"]


@dataclass(frozen=True)
class LangfusePromptSpec:
    name: str
    prompt: str
    labels: list[str]
    tags: list[str]
    config: dict[str, Any]
    commit_message: str


def agent_prompt_specs(
    *,
    agent: str = _DEFAULT_AGENT,
    config_path: Path | None = None,
) -> list[LangfusePromptSpec]:
    config = config_path or _AGENTS_DIR / agent / "config.yaml"
    executor_config = _executor_config(config)
    specs: list[LangfusePromptSpec] = []

    main_name = _config_text(executor_config, "instruction_prompt_name")
    main_prompt = _config_text(executor_config, "fallback_instruction")
    if main_name and main_prompt:
        specs.append(
            LangfusePromptSpec(
                name=main_name,
                prompt=main_prompt,
                labels=_labels(executor_config.get("instruction_prompt_label")),
                tags=[*_DEFAULT_TAGS, agent],
                config={"agent": agent, "prompt_role": "agent_instruction"},
                commit_message=f"Sync {agent} instruction from imp_agent_core config",
            )
        )

    sub_agent_tool = executor_config.get("sub_agent_tool")
    if isinstance(sub_agent_tool, dict):
        sub_name = _config_text(sub_agent_tool, "instruction_prompt_name")
        sub_prompt = _config_text(sub_agent_tool, "instruction_prompt")
        if sub_name and sub_prompt:
            specs.append(
                LangfusePromptSpec(
                    name=sub_name,
                    prompt=sub_prompt,
                    labels=_labels(sub_agent_tool.get("instruction_prompt_label")),
                    tags=[*_DEFAULT_TAGS, agent],
                    config={"agent": agent, "prompt_role": "sub_agent_instruction"},
                    commit_message=(
                        f"Sync {agent} sub-agent instruction from imp_agent_core config"
                    ),
                )
            )

    return specs


def zep_prompt_specs(config_path: Path | None = None) -> list[LangfusePromptSpec]:
    return agent_prompt_specs(agent="zep_agent", config_path=config_path)


def sync_prompts(
    specs: list[LangfusePromptSpec],
    *,
    dry_run: bool = False,
    client: Any | None = None,
) -> list[str]:
    synced: list[str] = []
    langfuse = client
    if langfuse is None and not dry_run:
        bootstrap_env()
        langfuse = get_client()

    for spec in specs:
        synced.append(spec.name)
        if dry_run:
            print(f"DRY RUN {spec.name} labels={spec.labels} chars={len(spec.prompt)}")
            continue
        langfuse.create_prompt(
            name=spec.name,
            prompt=spec.prompt,
            labels=spec.labels,
            tags=spec.tags,
            type="text",
            config=spec.config,
            commit_message=spec.commit_message,
        )
        print(f"Synced {spec.name} labels={spec.labels}")

    if langfuse is not None and hasattr(langfuse, "flush"):
        langfuse.flush()
    return synced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default=_DEFAULT_AGENT,
        help="Agent folder under agents/ that contains config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit agent config.yaml path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts that would be synced without calling Langfuse.",
    )
    args = parser.parse_args()

    specs = agent_prompt_specs(agent=args.agent, config_path=args.config)
    if not specs:
        config = args.config or _AGENTS_DIR / args.agent / "config.yaml"
        raise SystemExit(f"No prompt specs found in {config}")
    sync_prompts(specs, dry_run=args.dry_run)


def _executor_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    executor_config = raw.get("executor_config")
    if not isinstance(executor_config, dict):
        raise ValueError(f"executor_config must be a mapping: {config_path}")
    return executor_config


def _config_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    return str(value or "").strip()


def _labels(raw_label: Any) -> list[str]:
    label = str(raw_label or "").strip()
    return [label] if label else []


if __name__ == "__main__":
    main()
