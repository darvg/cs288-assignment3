from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from ..config_utils import load_runtime_config
from ..io_utils import ensure_parent_dir, read_jsonl, write_json
from ..pipeline.qa_pipeline import QAPipeline
from .metrics import exact_match, token_f1


def run_experiment(config: dict) -> dict:
    runtime = load_runtime_config("config/runtime.yaml")
    runtime["runtime"]["max_contexts"] = config.get("prompt_contexts", runtime["runtime"]["max_contexts"])
    runtime["runtime"]["model_name"] = config.get("model_name", runtime["runtime"]["model_name"])
    pipeline = QAPipeline(runtime)
    examples = read_jsonl("data/qa/qa_validation.jsonl")
    em_total = 0.0
    f1_total = 0.0
    timeout_count = 0
    for example in examples:
        pred = pipeline.answer_question(example["question"])
        if pred == "unknown":
            timeout_count += 1
        em_total += exact_match(pred, example["answers"])
        f1_total += token_f1(pred, example["answers"])
    count = max(len(examples), 1)
    return {
        "id": config["id"],
        "retriever": config["retriever"],
        "chunk_size": config["chunk_size"],
        "prompt_contexts": config["prompt_contexts"],
        "corpus": config["corpus"],
        "model_name": config["model_name"],
        "em": round(em_total / count, 4),
        "f1": round(f1_total / count, 4),
        "timeout_count": timeout_count,
        "latency_sec": round(0.0, 4),
        "notes": "Sample artifacts included; replace with full offline-built corpus for final runs.",
    }


def run_ablation_matrix(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = [run_experiment(experiment) for experiment in config["experiments"]]
    results_csv = config["defaults"]["results_csv"]
    ensure_parent_dir(results_csv)
    with open(results_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json("report_assets/intermediate/q3_system_summary.json", rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation_matrix(args.config)
