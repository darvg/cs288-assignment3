from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl


TITLE_SUFFIX_RE = re.compile(r"\s+(?:\||-)\s+EECS at (?:UC )?Berkeley$")
VERB_RULES = [
    ("wins", re.compile(r"^(.+?) wins (.+)$", re.I), "Who won {tail}?"),
    ("named", re.compile(r"^(.+?) named (.+)$", re.I), "Who was named {tail}?"),
    ("receives", re.compile(r"^(.+?) receives (.+)$", re.I), "Who received {tail}?"),
    ("elected", re.compile(r"^(.+?) elected (.+)$", re.I), "Who was elected {tail}?"),
    ("awarded", re.compile(r"^(.+?) awarded (.+)$", re.I), "Who was awarded {tail}?"),
    ("selected", re.compile(r"^(.+?) selected (.+)$", re.I), "Who was selected {tail}?"),
    ("appointed", re.compile(r"^(.+?) appointed (.+)$", re.I), "Who was appointed {tail}?"),
    ("joins", re.compile(r"^(.+?) joins (.+)$", re.I), "Who joins {tail}?"),
    ("honored", re.compile(r"^(.+?) honored (.+)$", re.I), "Who was honored {tail}?"),
    ("recognized", re.compile(r"^(.+?) recognized (.+)$", re.I), "Who was recognized {tail}?"),
]
CONTACT_RULES = [
    ("email", re.compile(r"\bEmail:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"), "What email address is listed on the {title} page?"),
    ("office", re.compile(r"\bOffice:\s*([^.\n]+)"), "What office is listed on the {title} page?"),
    ("located_in", re.compile(r"\blocated in\s+([^.\n]+)", re.I), "Where is {title} located?"),
]
RULE_QUOTAS = {
    "wins": 24,
    "named": 16,
    "receives": 10,
    "elected": 8,
    "awarded": 8,
    "selected": 8,
    "appointed": 6,
    "joins": 6,
    "honored": 6,
    "recognized": 5,
    "email": 0,
    "office": 0,
    "located_in": 0,
    "topic_resources": 12,
    "topic_academics": 10,
    "topic_research": 8,
    "topic_about": 4,
    "topic_people": 4,
    "topic_division": 4,
    "topic_department": 3,
    "topic_industry": 3,
    "topic_courses": 24,
    "topic_faculty": 18,
    "topic_pubs": 24,
}


def normalize_title(title: str) -> str:
    return TITLE_SUFFIX_RE.sub("", title).strip()


def section_for_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[0] if parts else "__root__"


def is_noise_page(url: str, title: str) -> bool:
    lower_title = title.lower()
    lower_url = url.lower()
    noisy_prefixes = {"category", "tag", "author", "book", "research_area"}
    section = section_for_url(url)
    if section in noisy_prefixes:
        return True
    if "/page/" in lower_url:
        return True
    if "archive" in lower_title and "colloquium" not in lower_url:
        return True
    if "archives -" in lower_title:
        return True
    return False


def descriptor_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    cleaned: list[str] = []
    for part in parts[1:]:
        if part.isdigit():
            continue
        words = [word for word in re.split(r"[-_]+", part) if word]
        cleaned.extend(words)
    descriptor = " ".join(cleaned[:8]).strip()
    return descriptor or "this topic"


def title_descriptor(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", normalize_title(title).lower()).split())


def course_code_from_title(title: str) -> str:
    match = re.match(r"^Course:\s+(.+)$", normalize_title(title))
    return " ".join(match.group(1).split()) if match else ""


def is_person_like_title(title: str) -> bool:
    cleaned = normalize_title(title)
    if len(cleaned.split()) < 2 or len(cleaned.split()) > 5:
        return False
    if ":" in cleaned or any(char.isdigit() for char in cleaned):
        return False
    return all(part[:1].isupper() for part in cleaned.split() if part)


def create_annotation_guidelines() -> str:
    return (
        "This benchmark is auto-generated from the live EECS crawl using high-confidence title and contact patterns. "
        "Review questions for phrasing quality, keep answers short, and preserve the source URL for retrieval evaluation."
    )


def validate_qa_record(record: dict) -> tuple[bool, str]:
    required = [
        "id",
        "question",
        "answers",
        "source_url",
        "page_title",
        "answer_type",
        "annotator",
        "section",
        "benchmark_rule",
    ]
    for key in required:
        if key not in record or record[key] in ("", []):
            return False, f"Missing field: {key}"
    return True, "ok"


def generate_candidates(pages: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    seen_questions: set[str] = set()
    for page in pages:
        title = normalize_title(page["title"])
        url = page["url"]
        if is_noise_page(url, title):
            continue
        section = section_for_url(url)
        added_for_page = False
        for rule_name, pattern, template in VERB_RULES:
            match = pattern.match(title)
            if not match:
                continue
            answer = " ".join(match.group(1).split())
            tail = " ".join(match.group(2).split())
            question = template.format(tail=tail).strip()
            key = f"{question}::{answer}"
            if key in seen_questions:
                break
            candidates.append(
                {
                    "question": question,
                    "answers": [answer],
                    "source_url": url,
                    "page_title": title,
                    "answer_type": "extractive",
                    "annotator": "auto",
                    "section": section,
                    "benchmark_rule": rule_name,
                    "confidence": "high",
                }
            )
            seen_questions.add(key)
            added_for_page = True
            break
        if added_for_page:
            continue
        for rule_name, pattern, template in CONTACT_RULES:
            match = pattern.search(page["text"])
            if not match:
                continue
            answer = " ".join(match.group(1).split())
            question = template.format(title=title).strip()
            key = f"{question}::{answer}"
            if key in seen_questions:
                break
            candidates.append(
                {
                    "question": question,
                    "answers": [answer],
                    "source_url": url,
                    "page_title": title,
                    "answer_type": "extractive",
                    "annotator": "auto",
                    "section": section,
                    "benchmark_rule": rule_name,
                    "confidence": "medium",
                }
            )
            seen_questions.add(key)
            break
        if section in {"resources", "academics", "research", "about", "people", "division", "department", "industry"}:
            descriptor = descriptor_from_url(url)
            question = f"Which EECS {section} page covers {descriptor}?"
            answer = title
            key = f"{question}::{answer}"
            if key not in seen_questions and len(answer.split()) <= 14:
                candidates.append(
                    {
                        "question": question,
                        "answers": [answer],
                        "source_url": url,
                        "page_title": title,
                        "answer_type": "extractive",
                        "annotator": "auto",
                        "section": section,
                        "benchmark_rule": f"topic_{section}",
                        "confidence": "medium",
                    }
                )
                seen_questions.add(key)
        if "/Courses/" in url:
            course_code = course_code_from_title(title)
            if course_code:
                question = f"Which EECS courses page covers {course_code.lower()}?"
                answer = title
                key = f"{question}::{answer}"
                if key not in seen_questions:
                    candidates.append(
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto",
                            "section": "courses",
                            "benchmark_rule": "topic_courses",
                            "confidence": "high",
                        }
                    )
                    seen_questions.add(key)
        if "/Faculty/Homepages/" in url and is_person_like_title(title):
            descriptor = title_descriptor(title)
            if descriptor:
                question = f"Which EECS faculty page covers {descriptor}?"
                answer = title
                key = f"{question}::{answer}"
                if key not in seen_questions:
                    candidates.append(
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto",
                            "section": "faculty",
                            "benchmark_rule": "topic_faculty",
                            "confidence": "high",
                        }
                    )
                    seen_questions.add(key)
        if "/Pubs/TechRpts/" in url:
            descriptor = title_descriptor(title)
            if descriptor and len(descriptor.split()) >= 4:
                question = f"Which EECS pubs page covers {descriptor}?"
                answer = title
                key = f"{question}::{answer}"
                if key not in seen_questions:
                    candidates.append(
                        {
                            "question": question,
                            "answers": [answer],
                            "source_url": url,
                            "page_title": title,
                            "answer_type": "extractive",
                            "annotator": "auto",
                            "section": "pubs",
                            "benchmark_rule": "topic_pubs",
                            "confidence": "high",
                        }
                    )
                    seen_questions.add(key)
    return candidates


def select_benchmark(candidates: list[dict], target_size: int) -> list[dict]:
    by_rule: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_rule[candidate["benchmark_rule"]].append(candidate)

    selected: list[dict] = []
    used_urls: set[str] = set()
    used_questions: set[str] = set()
    for rule_name, quota in RULE_QUOTAS.items():
        for candidate in by_rule.get(rule_name, []):
            if len([item for item in selected if item["benchmark_rule"] == rule_name]) >= quota:
                break
            if candidate["source_url"] in used_urls:
                continue
            if candidate["question"] in used_questions:
                continue
            selected.append(candidate)
            used_urls.add(candidate["source_url"])
            used_questions.add(candidate["question"])
            if len(selected) >= target_size:
                return assign_ids(selected)

    remaining = sorted(
        candidates,
        key=lambda item: (item["benchmark_rule"], item["section"], item["page_title"]),
    )
    for candidate in remaining:
        if len(selected) >= target_size:
            break
        if RULE_QUOTAS.get(candidate["benchmark_rule"], 1) == 0:
            continue
        if candidate["source_url"] in used_urls:
            continue
        if candidate["question"] in used_questions:
            continue
        selected.append(candidate)
        used_urls.add(candidate["source_url"])
        used_questions.add(candidate["question"])
    return assign_ids(selected[:target_size])


def assign_ids(records: list[dict]) -> list[dict]:
    output: list[dict] = []
    for index, record in enumerate(records, start=1):
        enriched = dict(record)
        enriched["id"] = f"live_q_{index:04d}"
        output.append(enriched)
    return output


def benchmark_stats(records: list[dict], candidates: list[dict]) -> dict:
    return {
        "num_candidates": len(candidates),
        "num_selected": len(records),
        "rule_counts": dict(Counter(record["benchmark_rule"] for record in records)),
        "section_counts": dict(Counter(record["section"] for record in records)),
        "guidelines": create_annotation_guidelines(),
    }


def review_subset(records: list[dict], size: int = 30) -> list[dict]:
    return records[:size]


def main(pages_path: str, out_path: str, target_size: int) -> None:
    pages = read_jsonl(pages_path)
    candidates = generate_candidates(pages)
    records = select_benchmark(candidates, target_size)
    for record in records:
        ok, message = validate_qa_record(record)
        if not ok:
            raise ValueError(message)
    write_jsonl(out_path, records)
    write_jsonl("data/qa/qa_live_benchmark_review.jsonl", review_subset(records))
    write_json(
        "report_assets/intermediate/live_benchmark_stats.json",
        benchmark_stats(records, candidates),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/interim/pages.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=140)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.pages, args.out, args.size)
