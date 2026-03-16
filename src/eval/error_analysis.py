from __future__ import annotations

from collections import Counter

from .metrics import exact_match


def categorize_error(record: dict) -> str:
    if record.get("em", 0.0) >= 1.0:
        return "correct"
    if record.get("retrieval_recall", 0.0) < 1.0:
        return "retrieval_miss"
    if record.get("pred", "") == "unknown":
        return "unsupported_or_timeout"
    if exact_match(record.get("pred", ""), record.get("golds", [])) == 0.0 and record.get("pred", ""):
        return "generation_mismatch"
    return "other"


def summarize_error_categories(records: list[dict]) -> dict:
    counter = Counter(categorize_error(record) for record in records)
    return dict(counter)
