"""
Measure OpenAI ChatGPT latency across several models (default: every id in DEFAULT_MODELS).

Modes:
  - plain: wall-clock time via perf_counter; prints a table.
  - langfuse: same calls through Langfuse-instrumented OpenAI SDK so latency
    (and usage) are recorded in Langfuse; still prints local timings.

Requires OPENAI_API_KEY. Langfuse mode also needs LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY (optional LANGFUSE_HOST for self-hosted).

Examples:
  python agent_speed_test.py
  python agent_speed_test.py --langfuse
  SPEED_TEST_MODELS=gpt-4o-mini,gpt-4o python agent_speed_test.py --query "what is 2+2"
  python agent_speed_test.py --models gpt-4o-mini,gpt-4o
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# All entries are used automatically unless SPEED_TEST_MODELS or --models overrides.
DEFAULT_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5",
    "gpt-5-Codex",
    "gpt-5.2-Codex",
    "gpt-5.3-Codex",
)
DEFAULT_QUERY = "what is 1+1"


@dataclass
class SpeedRow:
    model: str
    latency_s: float
    answer_preview: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None


def _parse_models(raw: str) -> list[str]:
    parts = [m.strip() for m in raw.replace(";", ",").split(",")]
    return [m for m in parts if m]


def _require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("Missing OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    return key


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    u = getattr(response, "usage", None)
    if u is None:
        return None, None, None
    return (
        getattr(u, "prompt_tokens", None),
        getattr(u, "completion_tokens", None),
        getattr(u, "total_tokens", None),
    )


def _chat_once(
    client: Any, model: str, user_text: str
) -> tuple[str, float, int | None, int | None, int | None]:
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        temperature=0.3,
    )
    elapsed = time.perf_counter() - t0
    text = ""
    if response.choices:
        msg = response.choices[0].message
        text = (msg.content or "").strip()
    pt, ct, tt = _extract_usage(response)
    return text, elapsed, pt, ct, tt


def run_plain(models: list[str], query: str) -> list[SpeedRow]:
    from openai import OpenAI

    api_key = _require_openai_key()
    client = OpenAI(api_key=api_key)
    rows: list[SpeedRow] = []
    for model in models:
        try:
            text, elapsed, pt, ct, tt = _chat_once(client, model, query)
            preview = text[:200] + ("…" if len(text) > 200 else "")
            rows.append(
                SpeedRow(
                    model=model,
                    latency_s=elapsed,
                    answer_preview=preview,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface API errors per model
            rows.append(
                SpeedRow(
                    model=model,
                    latency_s=0.0,
                    answer_preview="",
                    error=str(exc),
                )
            )
    return rows


def run_langfuse(models: list[str], query: str) -> list[SpeedRow]:
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not pk or not sk:
        print(
            "Langfuse mode requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    from langfuse import get_client
    from langfuse.openai import openai as lf_openai

    api_key = _require_openai_key()
    client = lf_openai.OpenAI(api_key=api_key)
    rows: list[SpeedRow] = []
    for model in models:
        try:
            text, elapsed, pt, ct, tt = _chat_once(client, model, query)
            preview = text[:200] + ("…" if len(text) > 200 else "")
            rows.append(
                SpeedRow(
                    model=model,
                    latency_s=elapsed,
                    answer_preview=preview,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                SpeedRow(
                    model=model,
                    latency_s=0.0,
                    answer_preview="",
                    error=str(exc),
                )
            )

    get_client().flush()
    return rows


def _tok(v: int | None) -> str:
    return "—" if v is None else str(v)


def print_table(rows: list[SpeedRow], title: str, query: str) -> None:
    print(title)
    print(f"User input: {query!r}")
    hdr = (
        f"{'model':<26} {'s':>8} {'in':>6} {'out':>6} {'total':>6}  "
        f"{'assistant_output_preview'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.error:
            print(
                f"{r.model:<26} {'—':>8} {'—':>6} {'—':>6} {'—':>6}  ERROR: {r.error}"
            )
        else:
            print(
                f"{r.model:<26} {r.latency_s:8.4f} "
                f"{_tok(r.prompt_tokens):>6} {_tok(r.completion_tokens):>6} "
                f"{_tok(r.total_tokens):>6}  {r.answer_preview!r}"
            )


def _resolve_models(cli_models: str | None) -> list[str]:
    if cli_models is not None:
        return _parse_models(cli_models)
    env = os.getenv("SPEED_TEST_MODELS", "").strip()
    if env:
        return _parse_models(env)
    return list(DEFAULT_MODELS)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM speed test across OpenAI chat models.")
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Use Langfuse-wrapped OpenAI client (records latency in Langfuse).",
    )
    parser.add_argument(
        "--query",
        default=os.getenv("SPEED_TEST_QUERY", DEFAULT_QUERY),
        help=f"User message (default: {DEFAULT_QUERY!r}).",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="M1,M2,...",
        help=(
            "Override model list (comma-separated). "
            f"Default: all {len(DEFAULT_MODELS)} models in DEFAULT_MODELS."
        ),
    )

    args = parser.parse_args()
    models = _resolve_models(args.models)
    if not models:
        print("No models to run (empty list).", file=sys.stderr)
        sys.exit(1)

    if args.langfuse:
        rows = run_langfuse(models, args.query)
        print_table(
            rows,
            "Langfuse-instrumented runs (also check Langfuse UI for traces)",
            args.query,
        )
    else:
        rows = run_plain(models, args.query)
        print_table(rows, "Plain OpenAI SDK (local timing only)", args.query)


if __name__ == "__main__":
    main()
