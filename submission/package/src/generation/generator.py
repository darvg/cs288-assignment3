from __future__ import annotations

import re
from typing import Any

import llm

from ..answer_postprocess import fallback_if_empty, sanitize_answer
from ..text_utils import simple_tokenize
from .prompts import build_qa_prompt


class AnswerGenerator:
    def __init__(self, model_name: str, timeout_sec: int):
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def generate(self, question: str, contexts: list[dict], use_llm: bool = False) -> str:
        if use_llm:
            prompt = build_qa_prompt(question, contexts)
            try:
                return fallback_if_empty(llm.generate(prompt, self.model_name, timeout_sec=self.timeout_sec))
            except Exception:
                return "unknown"
        return self._extractive_fallback(question, contexts)

    def _extractive_fallback(self, question: str, contexts: list[dict]) -> str:
        if not contexts:
            return "unknown"
        q_tokens = set(simple_tokenize(question))
        q_lower = question.lower()
        joined = " ".join(context["text"] for context in contexts)
        direct = _direct_answer_from_question(q_lower, joined)
        if direct:
            return fallback_if_empty(direct)
        best_answer = "unknown"
        best_score = float("-inf")
        for context in contexts:
            for candidate in _candidate_answers(context["text"]):
                score = len(q_tokens.intersection(simple_tokenize(candidate)))
                if "office" in q_lower and any(char.isdigit() for char in candidate):
                    score += 6
                if "email" in q_lower and "@" in candidate:
                    score += 6
                if "when" in q_lower and any(word in candidate.lower() for word in ["monday", "tuesday", "wednesday", "thursday", "friday", "am", "pm"]):
                    score += 6
                if "which building" in q_lower and any(char.isdigit() for char in candidate):
                    score += 6
                score -= 0.05 * len(candidate.split())
                if total_is_better(score, best_score, candidate, best_answer):
                    best_answer = candidate
                    best_score = score
        return fallback_if_empty(best_answer)


def total_is_better(score: float, best_score: float, candidate: str, best_answer: str) -> bool:
    return score > best_score or (score == best_score and len(candidate) < len(best_answer))


def _direct_answer_from_question(question: str, text: str) -> str:
    patterns = []
    if "which building" in question or "located" in question:
        patterns.append(r"\blocated in\b\s+(.+?)(?:\s+email:|$)")
    if "office number" in question or ("office" in question and "which building" not in question):
        patterns.append(r"\boffice\b\s*:\s*([^:]+?)(?:\s+email:|$)")
    if "email" in question:
        patterns.append(r"\bemail\b\s*:\s*(\S+)")
    if "when" in question:
        patterns.append(r"\bheld on\b\s+(.+?)(?:\s+in\s+.+|$)")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip(" .")
    return ""


def _candidate_answers(text: str) -> list[str]:
    sentences = [segment.strip(" .") for segment in re.split(r"[.;\n]", text) if segment.strip()]
    candidates: list[str] = []
    for sentence in sentences:
        office_match = re.search(r"\boffice\b\s*[:is]+\s*(.+?)(?:\s+email:|\s+$)", sentence, flags=re.I)
        if office_match:
            candidates.append(office_match.group(1).strip())
        located_match = re.search(r"\blocated in\b\s+(.+?)(?:\s+email:|\s+students\b|\s+$)", sentence, flags=re.I)
        if located_match:
            candidates.append(located_match.group(1).strip())
        email_match = re.search(r"\bemail\b\s*:\s*(\S+)", sentence, flags=re.I)
        if email_match:
            candidates.append(email_match.group(1).strip())
        held_match = re.search(r"\bheld on\b\s+(.+?)(?:\s+in\s+.+)?(?:\s+talks\b|\s+$)", sentence, flags=re.I)
        if held_match:
            candidates.append(held_match.group(1).strip())
        if ":" in sentence:
            tail = sentence.split(":", 1)[1].strip()
            if tail:
                candidates.append(tail)
        candidates.append(sentence)
    return [sanitize_answer(item) for item in candidates if item]
