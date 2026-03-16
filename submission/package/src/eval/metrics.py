from __future__ import annotations

import re
from collections import Counter


def normalize_answer(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def exact_match(pred: str, golds: list[str]) -> float:
    normalized_pred = normalize_answer(pred)
    return 1.0 if any(normalized_pred == normalize_answer(gold) for gold in golds) else 0.0


def token_f1(pred: str, golds: list[str]) -> float:
    pred_tokens = normalize_answer(pred).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if not overlap:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best
