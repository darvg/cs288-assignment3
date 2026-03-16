# Repository Guidelines

## Project Structure & Module Organization
This repository is intended to build a UC Berkeley EECS factoid QA RAG system from an empty scaffold. Keep runtime code in `src/`, offline data builders in `tools/`, YAML configs in `config/`, tests in `tests/`, and generated artifacts under `data/` and `report_assets/`. Follow the assignment layout from `instructions.md`: retrieval code belongs in `src/retrieval/`, generation code in `src/generation/`, pipeline orchestration in `src/pipeline/`, and evaluation code in `src/eval/`.

## Build, Test, and Development Commands
Use `python3` everywhere; the autograder entrypoint is `bash run.sh <questions_txt_path> <predictions_out_path>`.

`python3 tools/crawl_eecs.py --start-url https://eecs.berkeley.edu --out data/raw/html` crawls the source site offline.

`python3 tools/build_chunks.py --in data/interim/pages.jsonl --out data/interim/chunks.jsonl` creates retrieval chunks.

`python3 tools/build_bm25_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/bm25` and `python3 tools/build_dense_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/dense` build runtime retrieval artifacts.

`python3 -m src.eval.local_eval --questions data/qa/qa_validation.jsonl --predictions-out data/runs/predictions_dev.txt --results-out data/runs/experiment_results.csv` runs local evaluation.

`python3 tools/validate_submission.py` should be the final pre-submission check.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and small single-purpose modules. Prefer `snake_case` for files, functions, and variables; reserve `PascalCase` for classes like `QAPipeline` or retrievers. Keep answers newline-safe and route all model calls only through `llm.py`. Do not put expensive crawling or indexing inside `run.sh`.

## Testing Guidelines
Add focused tests in `tests/` with names like `test_io.py` and `test_smoke_run.py`. Cover I/O, answer postprocessing, metrics, and the autograder contract. The key smoke test is `bash run.sh data/sample/questions.txt data/sample/predictions.txt`, verifying one output line per input question and no embedded newlines.

## Commit & Pull Request Guidelines
This snapshot does not include local git history, so no project-specific convention can be inferred. Use short imperative commit subjects such as `Add hybrid retriever loader` or `Validate newline-safe predictions`. Keep commits scoped to one feature or fix. PRs should describe the user-visible impact, list commands run, note artifact/config changes, and include evaluation deltas for retrieval, EM, or F1 when behavior changes.

## Submission Constraints
Assume a CPU-only 4GB RAM runtime. Keep preprocessing offline, paths relative, and embedding models at or below the assignment limit. Package code, retrieval artifacts, and team-authored QA data together.
