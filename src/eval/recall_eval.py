from __future__ import annotations


def answer_string_recall(retrieved_texts: list[str], gold_answers: list[str]) -> float:
    haystack = " ".join(retrieved_texts).lower()
    return 1.0 if any(answer.lower() in haystack for answer in gold_answers if answer) else 0.0


def url_match_recall(retrieved_urls: list[str], gold_url: str) -> float:
    return 1.0 if gold_url and gold_url in retrieved_urls else 0.0
