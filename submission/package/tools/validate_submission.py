from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _contains_forbidden_openrouter_usage() -> bool:
    blocked_domain = "openrouter" + ".ai"
    blocked_env = "OPENROUTER" + "_API_KEY"
    for path in ROOT.rglob("*.py"):
        if path.name in {"llm.py", "validate_submission.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if blocked_domain in text or blocked_env in text:
            return True
    return False


def main() -> None:
    run_path = ROOT / "run.sh"
    _check(run_path.exists(), "run.sh is missing")
    mode = run_path.stat().st_mode
    _check(bool(mode & stat.S_IXUSR), "run.sh is not executable")
    _check((ROOT / "data/artifacts/bm25/metadata.jsonl").exists(), "BM25 artifacts are missing")
    _check((ROOT / "data/artifacts/dense/embeddings.npy").exists(), "Dense artifacts are missing")
    _check((ROOT / "data/qa/qa_validation.jsonl").exists(), "QA validation data is missing")
    _check(not _contains_forbidden_openrouter_usage(), "Forbidden OpenRouter usage outside llm.py")

    sample_questions = ROOT / "data/sample/questions.txt"
    sample_predictions = ROOT / "data/sample/validated_predictions.txt"
    subprocess.run(["bash", str(run_path), str(sample_questions), str(sample_predictions)], check=True, cwd=ROOT)
    questions = sample_questions.read_text(encoding="utf-8").splitlines()
    predictions = sample_predictions.read_text(encoding="utf-8").splitlines()
    _check(len(questions) == len(predictions), "Prediction line count mismatch")
    _check(all("\n" not in line for line in predictions), "Predictions contain embedded newlines")
    for path in ROOT.rglob("*.py"):
        if path.name == "validate_submission.py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"/home/|/Users/", text):
            raise SystemExit(f"Absolute local path found in {path}")
    print("Validation passed")


if __name__ == "__main__":
    main()
