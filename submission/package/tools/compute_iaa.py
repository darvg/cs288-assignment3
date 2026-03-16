from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import normalize_answer
from src.io_utils import read_jsonl, write_json


def exact_agreement(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)


def compute_iaa(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["id"], []).append(record)
    unique_questions = len(groups)
    compared = 0
    agreements = 0
    for items in groups.values():
        if len(items) < 2:
            continue
        compared += 1
        if exact_agreement(items[0]["answers"][0], items[1]["answers"][0]):
            agreements += 1
    return {
        "num_questions": unique_questions,
        "questions_with_double_annotation": compared,
        "double_annotation_fraction": round(compared / max(unique_questions, 1), 4),
        "exact_agreement": round(agreements / compared, 4) if compared else 0.0,
    }


def export_iaa_report(out_path: str, report: dict) -> None:
    write_json(out_path, report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    gold = read_jsonl(args.gold)
    subset = read_jsonl(args.subset)
    report = compute_iaa(gold + subset)
    export_iaa_report(args.out, report)
