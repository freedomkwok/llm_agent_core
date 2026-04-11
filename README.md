# imp_chat_agent

Python 3.11 Google ADK project scaffold for agent chat.

## Included setup

- Google ADK runtime (`google-adk`)
- Env loading with `python-dotenv`
- ADK root agent module in `agent.py`
- Basic dev tooling: `pytest`, `ruff`, `mypy`
- `Makefile` for quick setup/run/web/lint/test

## Quick start

1. Create Python 3.11 virtualenv and install:

```bash
make setup
```

2. Configure env:

```bash
cp .env.example .env
```

Set:

- `GOOGLE_API_KEY`
- `GOOGLE_MODEL` (default `gemini-2.5-flash`)

3. Run ADK CLI chat:

```bash
make run
```

4. Optional web UI:

```bash
make web
```

## Main files

- `agent.py`: ADK root agent (`root_agent`)
- `app.py`: simple bootstrap sanity check
- `requirements.txt`: runtime libraries
- `requirements-dev.txt`: dev libraries
- `pyproject.toml`: lint/test/typecheck config

## Development commands

```bash
make lint
make test
make typecheck
```
