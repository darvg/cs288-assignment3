# CS288 Assignment 3 RAG

This repository builds a retrieval-augmented QA system for UC Berkeley EECS factoid questions. It supports offline corpus preparation, local evaluation, ablations, report assets, and the required autograder runtime entrypoint:

```bash
bash run.sh <questions_txt_path> <predictions_out_path>
```

## Layout

- `src/`: runtime pipeline, retrieval, generation, evaluation
- `tools/`: offline crawling, corpus building, validation, packaging
- `config/`: runtime, retrieval, and experiment settings
- `data/`: raw pages, interim corpora, retrieval artifacts, QA data, runs
- `report_assets/`: report-ready tables and summaries
- `tests/`: unit tests and smoke test

## Offline preprocessing

```bash
python3 tools/crawl_eecs.py --start-url https://eecs.berkeley.edu --out data/raw/html
python3 tools/clean_html_to_text.py --in data/raw/html --out data/interim/pages.jsonl
python3 tools/build_chunks.py --in data/interim/pages.jsonl --out data/interim/chunks.jsonl
python3 tools/build_bm25_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/bm25
python3 tools/build_dense_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/dense
python3 tools/make_validation_set.py --out data/qa/qa_validation.jsonl
python3 tools/compute_iaa.py --gold data/qa/qa_validation.jsonl --subset data/qa/qa_iaa_subset.jsonl --out report_assets/intermediate/q1_dataset_stats.json
```

## Evaluation and ablations

```bash
python3 -m src.eval.local_eval --questions data/qa/qa_validation.jsonl --predictions-out data/runs/predictions_dev.txt --results-out data/runs/experiment_results.csv
python3 -m src.eval.ablations --config config/experiment.yaml
```

## Runtime

`run.sh` only loads prebuilt artifacts. It never crawls the site or rebuilds indexes. If `llm.py` is unavailable or LLM access is disabled, the pipeline falls back to extractive answer selection and returns `unknown` when unsupported.

If you run the repo on another machine and the runtime artifacts are missing, you can let it auto-download them by setting:

```bash
export RAG_CORPUS_BASE_URL="https://huggingface.co/datasets/<user>/<repo>/resolve/main"
```

The runtime will then fetch missing files under `data/artifacts/` and `data/interim/` before loading the pipeline.

## Validation

```bash
python3 tools/validate_submission.py
python3 -m unittest discover -s tests
```

## Optional Hugging Face Export

To publish the cleaned corpus to a Hugging Face dataset repo:

```bash
pip install -r requirements_export.txt
export HF_TOKEN=...
python3 tools/upload_cleaned_to_hf.py \
  --repo-id your-username/eecs-cleaned-corpus \
  --pages data/interim/pages.jsonl \
  --chunks data/interim/chunks.jsonl \
  --benchmark data/qa/qa_live_benchmark.jsonl
```

Use `--mode dataset` if you want `pages.jsonl` pushed as a dataset split rather than as raw files.

## Bootstrap assumption

This repository was designed for a blank-repo build. Sample data and sample retrieval artifacts are included so the runtime path works immediately and can be extended with the full EECS corpus offline.
