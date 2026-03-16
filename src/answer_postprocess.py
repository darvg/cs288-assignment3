from __future__ import annotations


def sanitize_answer(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def ensure_single_line(text: str) -> str:
    return sanitize_answer(text)


def fallback_if_empty(text: str) -> str:
    cleaned = sanitize_answer(text)
    return cleaned if cleaned else "unknown"
