from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bootstrap_artifacts import ensure_runtime_artifacts
from .config_utils import load_runtime_config
from .io_utils import read_questions_txt, write_answers_txt
from .pipeline.qa_pipeline import QAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    config = load_runtime_config(args.config)
    ensure_runtime_artifacts(config)
    pipeline = QAPipeline(config)
    questions = read_questions_txt(str(questions_path))
    answers = pipeline.answer_questions(questions)
    if len(answers) != len(questions):
        raise RuntimeError("Prediction count does not match input question count")
    write_answers_txt(args.output, answers)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        raise
