from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl


ALLOWED_URL_RE = re.compile(r"^https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?$", re.I)
TITLE_SUFFIX_RE = re.compile(r"\s+(?:\||-)\s+EECS at (?:UC )?Berkeley$")
EMAIL_RE = re.compile(r"\bEmail\b\s*:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", re.I)
PHONE_RE = re.compile(r"\bPhone\b\s*:\s*((?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]\d{4})", re.I)
OFFICE_RE = re.compile(r"\bOffice\b\s*:\s*([^:\n]+?)(?:\s+(?:Email|Phone|Office Hours)\b|$)", re.I)
OFFICE_HOURS_RE = re.compile(r"\bOffice Hours?\b\s*:\s*([^:\n]+?)(?:\s+(?:Email|Phone|Office)\b|$)", re.I)
DEADLINE_RE = re.compile(
    r"\b(?:Application|Registration|Submission)?\s*Deadline\b\s*:?\s*([^:\n]+?)(?:\s+(?:Apply Here|Categories|Related Articles)\b|$)",
    re.I,
)
PREREQ_RE = re.compile(
    r"\bPrerequisites?\b\s*:\s*([^:\n]+?)(?:\s+(?:Formats?|Credit Restrictions?|Grading Basis|Final Exam Status|Links?|Class Schedule)\b|$)",
    re.I,
)
UNITS_RE = re.compile(
    r"\bUnits?\b\s*:\s*([^:\n]+?)(?:\s+(?:Also Offered As|Prerequisites?|Formats?|Credit Restrictions?|Grading Basis|Final Exam Status|Links?|Class Schedule)\b|$)",
    re.I,
)
VERB_RULES = [
    ("wins", re.compile(r"^(.+?) wins (.+)$", re.I), ["Who won {tail}?", "Which person won {tail}?"]),
    ("named", re.compile(r"^(.+?) named (.+)$", re.I), ["Who was named {tail}?", "Which person was named {tail}?"]),
    ("receives", re.compile(r"^(.+?) receives (.+)$", re.I), ["Who received {tail}?", "Which person received {tail}?"]),
    ("elected", re.compile(r"^(.+?) elected (.+)$", re.I), ["Who was elected {tail}?", "Which person was elected {tail}?"]),
    ("awarded", re.compile(r"^(.+?) awarded (.+)$", re.I), ["Who was awarded {tail}?", "Which person was awarded {tail}?"]),
    ("appointed", re.compile(r"^(.+?) appointed (.+)$", re.I), ["Who was appointed {tail}?", "Which person was appointed {tail}?"]),
]
SECTION_TARGETS = {"resources", "academics", "research", "about", "people", "industry", "Courses", "Faculty", "news"}
RULE_LIMITS = {
    "course_prereq": 28,
    "course_units": 24,
    "contact_email": 24,
    "contact_phone": 12,
    "contact_office": 14,
    "contact_office_hours": 12,
    "deadline": 12,
    "news_person": 36,
}


def normalize_title(title: str) -> str:
    return TITLE_SUFFIX_RE.sub("", title).strip()


def is_allowed_url(url: str) -> bool:
    return bool(ALLOWED_URL_RE.match(url))


def section_for_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[0] if parts else "__root__"


def is_noise_page(url: str, title: str) -> bool:
    lower_title = title.lower()
    lower_url = url.lower()
    section = section_for_url(url)
    if section not in SECTION_TARGETS:
        return True
    if "/page/" in lower_url or "/category/" in lower_url or "/tag/" in lower_url or "/author/" in lower_url:
        return True
    if "redirecting" in lower_title:
        return True
    return False


def clean_answer(answer: str) -> str:
    answer = " ".join(answer.split())
    answer = re.sub(
        r"\s+(?:Formats?|Credit Restrictions?|Grading Basis|Final Exam Status|Links?|Class Schedule|Apply Here|Categories|Related Articles)\b.*$",
        "",
        answer,
        flags=re.I,
    )
    answer = re.sub(r"\s+", " ", answer).strip(" ,.;")
    return answer


def valid_answer(answer: str) -> bool:
    if not answer:
        return False
    if len(answer.split()) > 10:
        return False
    lowered = answer.lower()
    if any(marker in lowered for marker in ["skip to content", "expand main menu", "collapse main menu", "search for"]):
        return False
    return True


def course_code(title: str) -> str:
    match = re.match(r"^Course:\s+(.+)$", normalize_title(title))
    return " ".join(match.group(1).split()) if match else ""


def question_subject(title: str) -> str:
    cleaned = normalize_title(title)
    cleaned = re.sub(r"^Course:\s+", "", cleaned)
    cleaned = cleaned.replace("About About |", "").strip()
    return cleaned


def append_candidate(candidates: list[dict], seen: set[tuple[str, str]], record: dict) -> None:
    key = (record["question"], record["answers"][0])
    if key in seen:
        return
    seen.add(key)
    candidates.append(record)


def generate_candidates(pages: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for page in pages:
        url = page["url"]
        title = normalize_title(page["title"])
        text = page["text"]
        if not is_allowed_url(url) or is_noise_page(url, title):
            continue
        section = section_for_url(url)

        course = course_code(title)
        if course:
            prereq_match = PREREQ_RE.search(text)
            if prereq_match:
                answer = clean_answer(prereq_match.group(1))
                if valid_answer(answer):
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": f"What are the prerequisites for {course}?",
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": "courses",
                            "benchmark_rule": "course_prereq",
                        },
                    )
            units_match = UNITS_RE.search(text)
            if units_match:
                answer = clean_answer(units_match.group(1))
                if valid_answer(answer):
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": f"How many units is {course}?",
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": "courses",
                            "benchmark_rule": "course_units",
                        },
                    )

        email_match = EMAIL_RE.search(text)
        if email_match:
            answer = clean_answer(email_match.group(1))
            if valid_answer(answer):
                subject = question_subject(title)
                for question in [
                    f"What is the email for {subject}?",
                    f"What email address is listed on the {subject} page?",
                ]:
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": section,
                            "benchmark_rule": "contact_email",
                        },
                    )

        phone_match = PHONE_RE.search(text)
        if phone_match:
            answer = clean_answer(phone_match.group(1))
            if valid_answer(answer):
                subject = question_subject(title)
                for question in [
                    f"What phone number is listed on the {subject} page?",
                    f"What is the phone number for {subject}?",
                ]:
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": section,
                            "benchmark_rule": "contact_phone",
                        },
                    )

        office_match = OFFICE_RE.search(text)
        if office_match:
            answer = clean_answer(office_match.group(1))
            if valid_answer(answer):
                subject = question_subject(title)
                for question in [
                    f"What office is listed on the {subject} page?",
                    f"Where is the office for {subject}?",
                ]:
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": section,
                            "benchmark_rule": "contact_office",
                        },
                    )

        office_hours_match = OFFICE_HOURS_RE.search(text)
        if office_hours_match:
            answer = clean_answer(office_hours_match.group(1))
            if valid_answer(answer):
                subject = question_subject(title)
                for question in [
                    f"What office hours are listed on the {subject} page?",
                    f"When are the office hours for {subject}?",
                ]:
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": section,
                            "benchmark_rule": "contact_office_hours",
                        },
                    )

        deadline_match = DEADLINE_RE.search(text)
        if deadline_match:
            answer = clean_answer(deadline_match.group(1))
            if valid_answer(answer):
                subject = question_subject(title)
                for question in [
                    f"What is the deadline listed on the {subject} page?",
                    f"When is the deadline for {subject}?",
                ]:
                    append_candidate(
                        candidates,
                        seen,
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto_hidden_style",
                            "section": section,
                            "benchmark_rule": "deadline",
                        },
                    )

        for rule_name, pattern, templates in VERB_RULES:
            match = pattern.match(title)
            if not match:
                continue
            answer = clean_answer(match.group(1))
            tail = clean_answer(match.group(2))
            if not valid_answer(answer):
                continue
            for template in templates:
                append_candidate(
                    candidates,
                    seen,
                    {
                        "question": template.format(tail=tail),
                        "answers": [answer],
                        "source_url": url,
                        "page_title": title,
                        "answer_type": "extractive",
                        "annotator": "auto_hidden_style",
                        "section": section,
                        "benchmark_rule": "news_person",
                    },
                )
            break
    return candidates


def select_records(candidates: list[dict], target_size: int) -> list[dict]:
    by_rule: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_rule[candidate["benchmark_rule"]].append(candidate)

    selected: list[dict] = []
    used_urls: set[str] = set()
    used_questions: set[str] = set()
    for rule_name, limit in RULE_LIMITS.items():
        for candidate in by_rule.get(rule_name, []):
            if sum(1 for item in selected if item["benchmark_rule"] == rule_name) >= limit:
                break
            if candidate["question"] in used_questions:
                continue
            if candidate["source_url"] in used_urls and candidate["benchmark_rule"] not in {"course_prereq", "course_units"}:
                continue
            selected.append(candidate)
            used_questions.add(candidate["question"])
            used_urls.add(candidate["source_url"])
            if len(selected) >= target_size:
                return assign_ids(selected)

    for candidate in candidates:
        if len(selected) >= target_size:
            break
        limit = RULE_LIMITS.get(candidate["benchmark_rule"])
        if limit is not None and sum(1 for item in selected if item["benchmark_rule"] == candidate["benchmark_rule"]) >= limit:
            continue
        if candidate["question"] in used_questions:
            continue
        if candidate["source_url"] in used_urls and candidate["benchmark_rule"] not in {"course_prereq", "course_units"}:
            continue
        selected.append(candidate)
        used_questions.add(candidate["question"])
        used_urls.add(candidate["source_url"])
    return assign_ids(selected[:target_size])


def assign_ids(records: list[dict]) -> list[dict]:
    output = []
    for idx, record in enumerate(records, start=1):
        enriched = dict(record)
        enriched["id"] = f"hidden_style_q_{idx:04d}"
        output.append(enriched)
    return output


def stats(records: list[dict], candidates: list[dict]) -> dict:
    return {
        "num_candidates": len(candidates),
        "num_selected": len(records),
        "rule_counts": dict(Counter(record["benchmark_rule"] for record in records)),
        "section_counts": dict(Counter(record["section"] for record in records)),
        "note": "Natural-language hidden-dev-style validation set generated from single-page extractive EECS facts.",
    }


def main(pages_path: str, out_path: str, size: int) -> None:
    pages = read_jsonl(pages_path)
    candidates = generate_candidates(pages)
    records = select_records(candidates, size)
    write_jsonl(out_path, records)
    write_json("report_assets/intermediate/hidden_style_validation_stats.json", stats(records, candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/interim/pages.jsonl")
    parser.add_argument("--out", default="data/qa/qa_hidden_style_validation.jsonl")
    parser.add_argument("--size", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pages, args.out, args.size)
