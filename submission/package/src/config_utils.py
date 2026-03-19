from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def resolve_project_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def load_runtime_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(resolve_project_root()) / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = parse_simple_yaml(handle.read())
    config["resolved_project_root"] = resolve_project_root()
    return config


def parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = _preprocess_yaml_lines(text)
    if not lines:
        return {}
    parsed, next_index = _parse_block(lines, 0, lines[0][0])
    if next_index != len(lines):
        raise ValueError("Unparsed trailing YAML content")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    return parsed


def _preprocess_yaml_lines(text: str) -> list[tuple[int, str]]:
    processed: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        processed.append((indent, raw_line.strip()))
    return processed


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent, {})


def _parse_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
    seed: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    mapping = dict(seed)
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or content.startswith("- "):
            break
        key, sep, remainder = content.partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML mapping line: {content}")
        key = key.strip()
        remainder = remainder.strip()
        index += 1
        if remainder:
            mapping[key] = _parse_scalar(remainder)
            continue
        if index < len(lines) and lines[index][0] > current_indent:
            value, index = _parse_block(lines, index, lines[index][0])
            mapping[key] = value
        else:
            mapping[key] = {}
    return mapping, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break
        item_content = content[2:].strip()
        index += 1
        if not item_content:
            if index < len(lines) and lines[index][0] > current_indent:
                value, index = _parse_block(lines, index, lines[index][0])
                items.append(value)
            else:
                items.append(None)
            continue
        if ":" in item_content:
            item_mapping, index = _parse_mapping(
                lines,
                index,
                indent + 2,
                _mapping_seed_from_inline(item_content),
            )
            items.append(item_mapping)
            continue
        items.append(_parse_scalar(item_content))
    return items, index


def _mapping_seed_from_inline(content: str) -> dict[str, Any]:
    key, sep, remainder = content.partition(":")
    if not sep:
        raise ValueError(f"Invalid inline YAML mapping: {content}")
    remainder = remainder.strip()
    return {key.strip(): _parse_scalar(remainder) if remainder else {}}


def _parse_scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value
