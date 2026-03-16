from __future__ import annotations

import argparse
import csv
import time

from ..config_utils import load_runtime_config
from ..io_utils import read_jsonl, write_answers_txt
from ..pipeline.qa_pipeline import QAPipeline
from .error_analysis import categorize_error, summarize_error_categories
from .metrics import exact_match, token_f1
from .recall_eval import answer_string_recall, url_match_recall


def run_local_eval(questions_path: str, predictions_out: str, results_out: str) -> None:
    config = load_runtime_config("config/runtime.yaml")
    pipeline = QAPipeline(config)
    examples = read_jsonl(questions_path)
    predictions: list[str] = []
    records: list[dict] = []
    start = time.time()
    for example in examples:
        pred = pipeline.answer_question(example["question"])
        predictions.append(pred)
        retrieved = pipeline.hybrid.search(example["question"], top_k=config["runtime"]["top_k"])
        record = {
            "id": example["id"],
            "question": example["question"],
            "pred": pred,
            "golds": example["answers"],
            "em": exact_match(pred, example["answers"]),
            "f1": token_f1(pred, example["answers"]),
            "retrieval_recall": answer_string_recall([chunk.text for chunk in retrieved], example["answers"]),
            "url_recall": url_match_recall([chunk.url for chunk in retrieved], example.get("source_url", "")),
        }
        record["category"] = categorize_error(record)
        records.append(record)
    elapsed = time.time() - start
    write_answers_txt(predictions_out, predictions)
    with open(results_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "question", "pred", "em", "f1", "retrieval_recall", "url_recall", "category"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})
    summary_path = results_out.rsplit(".", 1)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        mean_em = sum(record["em"] for record in records) / max(len(records), 1)
        mean_f1 = sum(record["f1"] for record in records) / max(len(records), 1)
        handle.write(f"examples={len(records)}\n")
        handle.write(f"em={mean_em:.4f}\n")
        handle.write(f"f1={mean_f1:.4f}\n")
        handle.write(f"latency_sec={elapsed:.4f}\n")
        handle.write(f"categories={summarize_error_categories(records)}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--results-out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_local_eval(args.questions, args.predictions_out, args.results_out)
