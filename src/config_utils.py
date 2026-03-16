from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_project_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def load_runtime_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(resolve_project_root()) / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["resolved_project_root"] = resolve_project_root()
    return config
