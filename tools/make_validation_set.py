from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import write_json, write_jsonl


def create_annotation_guidelines() -> str:
    return (
        "Write short factoid questions answerable from a single EECS page. "
        "Prefer extractive answers under 10 words, include the source URL, and use 'unknown' only when unsupported."
    )


def validate_qa_record(record: dict) -> tuple[bool, str]:
    required = ["id", "question", "answers", "source_url", "page_title", "answer_type", "annotator"]
    for key in required:
        if key not in record or record[key] in ("", []):
            return False, f"Missing field: {key}"
    return True, "ok"


def export_validation_template(out_path: str) -> None:
    template = [
        {
            "id": "q_0001",
            "question": "What is the office number of Dan Klein?",
            "answers": ["773 Soda Hall"],
            "source_url": "https://eecs.berkeley.edu/people/faculty/dan-klein/",
            "page_title": "Dan Klein",
            "answer_type": "extractive",
            "annotator": "A",
        }
    ]
    write_jsonl(out_path, template)


def _default_records() -> list[dict]:
    templates = [
        (
            [
                "What is the office number of Dan Klein?",
                "Which office is listed for Dan Klein?",
                "Where is Dan Klein's office?",
                "What room is Dan Klein in?",
                "Which Soda Hall office belongs to Dan Klein?",
            ],
            ["773 Soda Hall"],
            "https://eecs.berkeley.edu/people/faculty/dan-klein/",
            "Dan Klein",
        ),
        (
            [
                "Which building houses the EECS undergraduate advising office?",
                "Where is the undergraduate advising office located?",
                "What room is listed for EECS undergraduate advising?",
                "What is the advising office location?",
                "Where can students find undergraduate advising?",
            ],
            ["354 Cory Hall"],
            "https://eecs.berkeley.edu/resources/undergrads/advising/",
            "Undergraduate Advising",
        ),
        (
            [
                "When is the department seminar held?",
                "What time is the EECS seminar?",
                "On what schedule does the department seminar meet?",
                "When does the EECS seminar take place?",
                "What is the seminar time?",
            ],
            ["Fridays at 4pm"],
            "https://eecs.berkeley.edu/events/seminar/",
            "EECS Seminar",
        ),
        (
            [
                "What is the email for undergraduate advising?",
                "Which email address should students use for undergraduate advising?",
                "How do you email undergraduate advising?",
                "What contact email is listed for the advising office?",
                "What is the advising office email?",
            ],
            ["ugrad@eecs.berkeley.edu"],
            "https://eecs.berkeley.edu/resources/undergrads/advising/",
            "Undergraduate Advising",
        ),
    ]
    records: list[dict] = []
    question_id = 1
    while len(records) < 100:
        for questions, answers, source_url, page_title in templates:
            for question in questions:
                if len(records) >= 100:
                    break
                records.append(
                    {
                        "id": f"q_{question_id:04d}",
                        "question": question,
                        "answers": answers,
                        "source_url": source_url,
                        "page_title": page_title,
                        "answer_type": "extractive",
                        "annotator": "A",
                    }
                )
                question_id += 1
            if len(records) >= 100:
                break
    return records


def _iaa_subset(records: list[dict]) -> list[dict]:
    subset: list[dict] = []
    for record in records[:30]:
        copy = dict(record)
        copy["annotator"] = "B"
        subset.append(copy)
    return subset


def _run(out_path: str) -> None:
    records = _default_records()
    for record in records:
        ok, message = validate_qa_record(record)
        if not ok:
            raise ValueError(message)
    write_jsonl(out_path, records)
    subset_path = "data/qa/qa_iaa_subset.jsonl"
    write_jsonl(subset_path, _iaa_subset(records))
    write_json(
        "report_assets/intermediate/q1_examples.json",
        {"guidelines": create_annotation_guidelines(), "examples": records[:3]},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run(args.out)
