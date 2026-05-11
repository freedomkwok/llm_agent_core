# SPDX-License-Identifier: Apache-2.0
.PHONY: setup run lint test typecheck build-llm

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt

# Build wheel/sdist of llm_inference_core (output: …/llm_inference_core/dist/).
LLM_INFERENCE_CORE := /Users/freedomkwokmacbookpro/Github/imp/llm_inference_core
build-llm:
	cd $(LLM_INFERENCE_CORE) && python3 -m pip install -q build && python3 -m build

run:
	.venv/bin/adk run .

web:
	.venv/bin/adk web .

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest -q

typecheck:
	.venv/bin/mypy app.py agent.py
