from __future__ import annotations

import re

import llm

from ..answer_postprocess import fallback_if_empty, sanitize_answer
from ..text_utils import simple_tokenize
from .prompts import build_qa_prompt

TITLE_SUFFIX_RE = re.compile(r"\s+(?:\||-)\s+EECS at (?:UC )?Berkeley$")
TITLE_VERB_PATTERNS = [
    ("who won ", re.compile(r"^(.+?) wins (.+)$", re.I)),
    ("who was named ", re.compile(r"^(.+?) named (.+)$", re.I)),
    ("who received ", re.compile(r"^(.+?) receives (.+)$", re.I)),
    ("who was elected ", re.compile(r"^(.+?) elected (.+)$", re.I)),
    ("who was awarded ", re.compile(r"^(.+?) awarded (.+)$", re.I)),
    ("who was selected ", re.compile(r"^(.+?) selected (.+)$", re.I)),
    ("who was appointed ", re.compile(r"^(.+?) appointed (.+)$", re.I)),
    ("who joins ", re.compile(r"^(.+?) joins (.+)$", re.I)),
    ("who was honored ", re.compile(r"^(.+?) honored (.+)$", re.I)),
    ("who was recognized ", re.compile(r"^(.+?) recognized (.+)$", re.I)),
]
TEXT_VERB_PATTERNS = [
    ("who won ", ["wins", "has won", "won", "has been awarded", "was awarded"]),
    ("who was named ", ["named"]),
    ("who received ", ["receives", "received"]),
    ("who was elected ", ["elected"]),
    ("who was awarded ", ["awarded", "has been awarded", "was awarded"]),
    ("who was selected ", ["selected"]),
    ("who was appointed ", ["appointed"]),
    ("who joins ", ["joins", "joined"]),
    ("who was honored ", ["honored"]),
    ("who was recognized ", ["recognized"]),
]
TOPIC_PREFIXES = {
    "which eecs resources page covers ": "resources",
    "which eecs academics page covers ": "academics",
    "which eecs research page covers ": "research",
    "which eecs about page covers ": "about",
    "which eecs people page covers ": "people",
    "which eecs division page covers ": "division",
    "which eecs department page covers ": "department",
    "which eecs industry page covers ": "industry",
    "which eecs courses page covers ": "courses",
    "which eecs faculty page covers ": "faculty",
    "which eecs pubs page covers ": "pubs",
}


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
        title_direct = _direct_answer_from_context_titles(q_lower, contexts)
        if title_direct:
            return fallback_if_empty(title_direct)
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
        patterns.append(r"\blocated in\b\s+(.+?)(?:\.\s+|\s+email:|$)")
        patterns.append(r"\blocated at\b\s+(.+?)(?:\.\s+|\s+email:|$)")
    if "office number" in question or ("office" in question and "which building" not in question):
        patterns.append(r"\boffice\b\s*:\s*([^:]+?)(?:\.\s+|\s+email:|$)")
    if "email" in question:
        patterns.append(r"\bemail\b\s*:\s*(\S+)")
    if "when" in question:
        patterns.append(r"\bheld on\b\s+(.+?)(?:\s+in\s+.+|$)")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip(" .")
    return ""


def _direct_answer_from_context_titles(question: str, contexts: list[dict]) -> str:
    for prefix, pattern in TITLE_VERB_PATTERNS:
        if not question.startswith(prefix):
            continue
        target = _normalize_match_text(question[len(prefix) :])
        best_subject = ""
        best_score = -1
        for context in contexts:
            title = _normalize_title(context.get("title", ""))
            match = pattern.match(title)
            if not match:
                continue
            subject = sanitize_answer(match.group(1))
            tail = _normalize_match_text(match.group(2))
            score = _token_overlap(target, tail)
            if score > best_score:
                best_subject = subject
                best_score = score
        if best_subject:
            return best_subject

    for prefix, expected_section in TOPIC_PREFIXES.items():
        if not question.startswith(prefix):
            continue
        target = _normalize_match_text(question[len(prefix) :])
        best_title = ""
        best_score = -1
        for context in contexts:
            title = sanitize_answer(_normalize_title(context.get("title", "")))
            url = context.get("url", "").lower()
            if expected_section == "faculty":
                section_match = "/faculty/" in url
            elif expected_section == "courses":
                section_match = "/courses/" in url
            elif expected_section == "pubs":
                section_match = "/pubs/" in url
            else:
                section_match = f"/{expected_section}" in url
            if not section_match:
                continue
            score = _token_overlap(target, _normalize_match_text(title + " " + url))
            if score > best_score:
                best_title = title
                best_score = score
        if best_title:
            return best_title
    text_direct = _direct_answer_from_context_text(question, contexts)
    if text_direct:
        return text_direct
    return ""


def _normalize_title(title: str) -> str:
    return TITLE_SUFFIX_RE.sub("", title).strip()


def _normalize_match_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def _token_overlap(left: str, right: str) -> int:
    return len(set(left.split()).intersection(right.split()))


def _direct_answer_from_context_text(question: str, contexts: list[dict]) -> str:
    for prefix, verbs in TEXT_VERB_PATTERNS:
        if not question.startswith(prefix):
            continue
        target = question[len(prefix) :].rstrip("?").strip()
        candidate_matches: list[tuple[int, str]] = []
        for context in contexts:
            text = sanitize_answer(context.get("text", ""))
            for verb in verbs:
                for match in _subject_matches_for_target(text, verb, target):
                    candidate_matches.append((_token_overlap(_normalize_match_text(target), _normalize_match_text(text)), match))
        if candidate_matches:
            candidate_matches.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
            return candidate_matches[0][1]
    return ""


def _subject_matches_for_target(text: str, verb: str, target: str) -> list[str]:
    escaped_target = re.escape(target).replace(r"\ ", r"\s+")
    escaped_verb = re.escape(verb).replace(r"\ ", r"\s+")
    patterns = [
        rf"([A-Z][A-Za-z0-9.&'/-]+(?:\s+[A-Z][A-Za-z0-9.&'(),/-]+){{0,7}})\s+{escaped_verb}\s+(?:the\s+)?{escaped_target}",
        rf"([A-Z][A-Za-z0-9.&'/-]+(?:\s+[A-Z][A-Za-z0-9.&'(),/-]+){{0,7}})\s+{escaped_verb}.*?(?:the\s+)?{escaped_target}",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for found in re.finditer(pattern, text, flags=re.I):
            subject = _clean_subject(found.group(1))
            if subject:
                matches.append(subject)
    return matches


def _clean_subject(subject: str) -> str:
    subject = sanitize_answer(subject)
    subject = re.sub(r"^(Professor Emeritus|Professor|Prof\.?|Assistant Professor|Associate Professor|EECS Professor|EECS Prof\.?)\s+", "", subject, flags=re.I)
    subject = re.sub(r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\s+", "", subject)
    subject = re.sub(r"^\d{4}\s+", "", subject)
    subject = re.sub(r"\([^)]*\)", "", subject).strip(" ,.-")
    parts = [part for part in re.split(r"\s+", subject) if part]
    if not parts:
        return ""
    if len(parts) > 6:
        parts = parts[-6:]
    return sanitize_answer(" ".join(parts))


def _candidate_answers(text: str) -> list[str]:
    sentences = [segment.strip(" .") for segment in re.split(r"[.;\n]", text) if segment.strip()]
    candidates: list[str] = []
    for sentence in sentences:
        office_match = re.search(r"\boffice\b\s*[:is]+\s*(.+?)(?:\s+email:|\s+$)", sentence, flags=re.I)
        if office_match:
            candidates.append(office_match.group(1).strip())
        located_match = re.search(r"\blocated in\b\s+(.+?)(?:\.\s+|\s+email:|\s+students\b|\s+$)", sentence, flags=re.I)
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
