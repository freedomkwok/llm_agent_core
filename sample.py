import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "registry.yaml"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def load_registry() -> list[dict]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["skills"]


def load_skill_prompt(load_path: str) -> str:
    path = BASE_DIR / load_path
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_router_prompt(user_input: str, skills: list[dict]) -> str:
    compact = []
    for skill in skills:
        compact.append(
            {
                "id": skill["id"],
                "title": skill["title"],
                "summary": skill["summary"],
                "triggers": skill.get("triggers", []),
                "stage": skill.get("stage", []),
                "tools": skill.get("tools", []),
            }
        )

    return f"""
        你是一个 skill router。
        你的任务不是回答用户，而是从候选 skill 中选出最适合的一个。

        要求：
        1. 只返回 JSON
        2. JSON 格式必须为:
        {{
        "selected_skill": "...",
        "reason": "...",
        "confidence": 0.0
        }}

        候选 skills:
    {json.dumps(compact, ensure_ascii=False, indent=2)}

    用户请求:
    {user_input}
    """.strip()


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if isinstance(parts, list):
            out = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    out.append(part_text)
            if out:
                return "\n".join(out).strip()
    return ""


def _generate(client: genai.Client, model: str, prompt: str) -> str:
    response = client.models.generate_content(model=model, contents=prompt)
    text = _extract_text(response)
    if not text:
        raise RuntimeError("Model returned empty output.")
    return text


def select_skill(client: genai.Client, model: str, user_input: str, skills: list[dict]) -> dict:
    router_prompt = build_router_prompt(user_input, skills)
    combined_prompt = (
        "你是严格的 skill router，只输出 JSON，不要解释。\n\n"
        f"{router_prompt}"
    )
    text = _generate(client, model, combined_prompt)

    try:
        picked = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Router output is not valid JSON: {text}") from exc

    selected_id = picked["selected_skill"]
    for skill in skills:
        if skill["id"] == selected_id:
            return skill

    raise RuntimeError(f"Selected skill not found: {selected_id}")


def answer_with_skill(client: genai.Client, model: str, user_input: str, skill: dict) -> str:
    skill_prompt = load_skill_prompt(skill["load_path"])
    prompt = f"""
你是一个多技能助手。
现在你只启用以下 skill，请严格按照它工作。

{skill_prompt}

用户请求：
{user_input}
""".strip()
    return _generate(client, model, prompt)


def main() -> None:
    load_dotenv(BASE_DIR / ".env")


    skills = load_registry()

    print("请输入用户请求：")
    user_input = input("> ").strip()
    if not user_input:
        raise RuntimeError("User input cannot be empty.")

    selected_skill = select_skill(client, model, user_input, skills)
    print("\n[Router Selected]")
    print(json.dumps(selected_skill, ensure_ascii=False, indent=2))

    final_answer = answer_with_skill(client, model, user_input, selected_skill)
    print("\n[Final Answer]")
    print(final_answer)


if __name__ == "__main__":
    main()