from pathlib import Path

import yaml


def test_registry_has_skills() -> None:
    root = Path(__file__).resolve().parent.parent
    registry_path = root / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert "skills" in data
    assert isinstance(data["skills"], list)
    assert len(data["skills"]) > 0
