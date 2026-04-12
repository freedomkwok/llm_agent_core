"""Step 2: define an ADK LlmAgent + tool."""

import os
from contextlib import nullcontext

import requests
from google.adk.agents import LlmAgent
from langfuse import get_client


def _start_observation(
    *,
    name: str,
    as_type: str = "span",
    input_data: dict | None = None,
    metadata: dict | None = None,
):
    """Best-effort Langfuse observation context manager."""
    client = get_client()
    kwargs = {
        "name": name,
        "as_type": as_type,
        "input": input_data or {},
        "metadata": metadata or {},
    }

    trace_id = os.getenv("A2A_TRACE_ID", "").strip()
    parent_observation_id = os.getenv("A2A_PARENT_OBSERVATION_ID", "").strip()
    if trace_id:
        kwargs["trace_id"] = trace_id
    if parent_observation_id:
        kwargs["parent_observation_id"] = parent_observation_id

    try:
        return client.start_as_current_observation(**kwargs)
    except Exception:
        return nullcontext()


def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
) -> dict:
    """Query Frankfurter API for exchange rates."""
    with _start_observation(
        name="tool_get_exchange_rate",
        as_type="tool",
        input_data={
            "currency_from": currency_from,
            "currency_to": currency_to,
            "currency_date": currency_date,
        },
    ) as obs:
        try:
            response = requests.get(
                f"https://api.frankfurter.app/{currency_date}",
                params={"from": currency_from, "to": currency_to},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            if hasattr(obs, "update"):
                obs.update(output=payload)
            return payload
        except requests.RequestException as exc:
            error_payload = {"error": str(exc)}
            if hasattr(obs, "update"):
                obs.update(output=error_payload)
            return error_payload


with _start_observation(
    name="build_my_llm_agent",
    as_type="span",
    input_data={"model": "gemini-2.0-flash"},
):
    pass

my_llm_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="currency_exchange_agent",
    description="An agent that can provide currency exchange rates.",
    instruction=(
        "You are a helpful currency exchange assistant. "
        "Use get_exchange_rate tool to answer user questions. "
        "If the tool fails, explain the error."
    ),
    tools=[get_exchange_rate],
)


if __name__ == "__main__":
    print(my_llm_agent.name)
