# CS288 Assignment 3 — Agentic End-to-End Build Spec for Codex

You are building this project from a completely empty repository. There is no starter code. Treat repository creation as part of the required work, not as a preliminary convenience step.

Your job is to create a complete, self-contained repository for a Retrieval-Augmented Generation system that answers factoid questions about UC Berkeley EECS using only information from `eecs.berkeley.edu`. The final repo must support offline data preparation, local evaluation, ablations, report support, and autograder execution through a required `run.sh` entrypoint. The early milestone uses your own retrieval corpus and validation data; the final phase also uses the released dev set and reference retrieval corpus. 

The task is factoid QA over English EECS pages, answerable from a single page, with short answers usually under 10 words. Evidence must come from plain text or tables, not images or PDFs. A small fraction of yes/no and short abstractive questions is allowed. 

The autograder contract is strict: it will run `bash run.sh <questions_txt_path> <predictions_out_path>`. Your code must use `python3`, write exactly one answer per input line in the same order, avoid embedded newlines, keep paths relative, and stay within a CPU-only 4GB RAM environment. The embedding model must be 400MB or smaller. Only the allowed libraries may be assumed in the runtime environment. 

All LLM calls must go through the provided `llm.py`. Do not modify it. Do not call OpenRouter from any other file. If runtime inference is attempted before `llm.py` is present, fail clearly. The allowed generator models are restricted to the listed model names in the assignment.  

The written report must cover QA data creation, retrieval corpus construction, RAG system design, two ablations, error analysis including false negatives due to metric limitations, and takeaways/future ideas. It must also include a contribution statement, GenAI statement, and references, outside the main page limit.

---

## 1. Mission

Build the repo end-to-end so that a blank directory becomes a valid submission repository with:

* working repo structure
* offline corpus and index builders
* local QA dataset tooling
* runtime RAG inference path
* evaluation and ablation scripts
* submission validation and packaging
* report-support outputs

Success means all of the following are true:

1. `bash run.sh <questions_txt_path> <predictions_out_path>` works with prebuilt artifacts. 
2. The repo contains code plus retrieval datastore/model artifacts plus the QA data created by the team. 
3. Runtime uses only allowed dependencies and only `llm.py` for model calls. 
4. The repo leaves behind enough outputs to write the report sections required by the assignment.

---

## 2. Non-negotiable operating rules

### 2.1 Do not improvise around the autograder

The required runtime interface is fixed. `run.sh` must exist with that exact name, accept exactly two positional arguments, use `python3`, and write one answer per line to the provided output path. 

### 2.2 Do not do heavy work at runtime

Runtime must not crawl the site, rebuild indexes, or do expensive preprocessing. That work belongs in offline tooling only. The assignment allows offline work, but runtime should not access the internet other than via the provided `llm.py` wrapper for allowed LLM calls.

### 2.3 Do not violate the `llm.py` boundary

All generator calls must be routed through `llm.py`. Never modify it. Never replace it. Never bypass it. The autograder checks this.

### 2.4 Optimize for robustness, not novelty

Prefer simple, deterministic, low-risk implementations that fit CPU-only 4GB RAM and finish well within the time limit. The assignment explicitly warns about timeouts and recommends timeout handling per question.

---

## 3. Required repository structure

Create exactly this layout or a functionally equivalent one:

```text
project_root/
├── .gitignore
├── README.md
├── run.sh
├── llm.py
├── requirements_runtime.txt
├── requirements_offline.txt
├── config/
│   ├── runtime.yaml
│   ├── retrieval.yaml
│   └── experiment.yaml
├── src/
│   ├── __init__.py
│   ├── main_predict.py
│   ├── io_utils.py
│   ├── text_utils.py
│   ├── answer_postprocess.py
│   ├── config_utils.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrieval_types.py
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── hybrid_retriever.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── generator.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── qa_pipeline.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py
│       ├── local_eval.py
│       ├── recall_eval.py
│       ├── ablations.py
│       └── error_analysis.py
├── tools/
│   ├── crawl_eecs.py
│   ├── clean_html_to_text.py
│   ├── build_corpus.py
│   ├── build_chunks.py
│   ├── build_bm25_index.py
│   ├── build_dense_index.py
│   ├── make_validation_set.py
│   ├── compute_iaa.py
│   ├── validate_submission.py
│   ├── package_submission.py
│   └── report_tables.py
├── tests/
│   ├── test_io.py
│   ├── test_postprocess.py
│   ├── test_metrics.py
│   └── test_smoke_run.py
├── data/
│   ├── sample/
│   ├── raw/
│   │   └── html/
│   ├── interim/
│   ├── artifacts/
│   │   ├── bm25/
│   │   └── dense/
│   ├── qa/
│   └── runs/
├── report_assets/
│   ├── figures/
│   ├── tables/
│   └── intermediate/
└── submission/
```

Every file you create must either be executable, importable, or intentionally data-only with clear purpose.

---

## 4. Build strategy

Use two layers.

### Layer A: offline asset-building layer

This layer handles crawling, HTML cleaning, corpus assembly, chunking, embedding, index construction, QA dataset creation, IAA computation, ablations, and report artifacts.

### Layer B: runtime inference layer

This layer loads prebuilt artifacts, reads question lines, retrieves chunks, calls the generator through `llm.py`, postprocesses answers, and writes one answer per line.

Do not let `run.sh` trigger Layer A.

---

## 5. Default technical design

Use a chunk-based hybrid RAG pipeline.

* Chunk size: 160 words
* Overlap: 40 words
* Sparse retrieval: BM25 via `rank-bm25`
* Dense retrieval: `sentence-transformers/all-MiniLM-L6-v2`
* Dense index: normalized embeddings with FAISS `IndexFlatIP`
* Fusion: reciprocal rank fusion
* Retrieve top 6 fused chunks
* Prompt with top 3 or 4 chunks
* Generator default: `qwen/qwen-2.5-7b-instruct`
* One LLM call per question
* Timeout per question
* Fallback answer on failure: `unknown`

These defaults are chosen to fit the assignment’s dependency limits, embedding size cap, CPU-only environment, and latency constraints.

---

## 6. Execution phases and stop/go gates

## Phase 0 — bootstrap the repository

### Objective

Turn a blank directory into a functioning Python repo with the required runtime entrypoint.

### Files to create first

1. `.gitignore`
2. `README.md`
3. `run.sh`
4. `requirements_runtime.txt`
5. `requirements_offline.txt`
6. `config/runtime.yaml`
7. `src/__init__.py`
8. `src/io_utils.py`
9. `src/config_utils.py`
10. `src/main_predict.py`
11. `src/pipeline/__init__.py`
12. `src/pipeline/qa_pipeline.py`

### Required behavior

Create a minimal smoke path that can:

* read sample questions
* produce placeholder one-line answers
* write the output file
* prove line count and formatting are correct

### Stop/go gate

Do not continue until this works:

```bash
bash run.sh data/sample/questions.txt data/sample/predictions.txt
```

and the output file has exactly the same number of lines as the input file. 

---

## Phase 1 — harden the runtime contract

### Objective

Make the required `run.sh` path reliable before building the full system.

### Required `run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

python3 -m src.main_predict \
  --questions "${QUESTIONS_PATH}" \
  --output "${PREDICTIONS_PATH}" \
  --config "config/runtime.yaml"
```

### Required functions

In `src/io_utils.py` implement:

```python
def read_questions_txt(path: str) -> list[str]: ...
def write_answers_txt(path: str, answers: list[str]) -> None: ...
```

In `src/config_utils.py` implement:

```python
def load_runtime_config(path: str) -> dict: ...
def resolve_project_root() -> str: ...
```

In `src/main_predict.py` implement:

```python
def parse_args() -> object: ...
def main() -> None: ...
```

### Stop/go gate

Do not proceed until:

* argument validation works
* paths are relative-safe
* missing artifact errors are readable
* output answers never contain newlines

---

## Phase 2 — create the local QA dataset

### Objective

Build your own validation set for development and for the report.

The assignment recommends at least 100 questions and at least 30% double annotation for IAA. The questions should follow the same scope as the hidden sets.

### Required files

* `tools/make_validation_set.py`
* `tools/compute_iaa.py`

### Required data format

`data/qa/qa_validation.jsonl` records should look like:

```json
{
  "id": "q_0001",
  "question": "What is the office number of Dan Klein?",
  "answers": ["773 Soda Hall"],
  "source_url": "https://eecs.berkeley.edu/...",
  "page_title": "...",
  "answer_type": "extractive",
  "annotator": "A"
}
```

### Required functions

In `tools/make_validation_set.py` implement:

```python
def create_annotation_guidelines() -> str: ...
def validate_qa_record(record: dict) -> tuple[bool, str]: ...
def export_validation_template(out_path: str) -> None: ...
```

In `tools/compute_iaa.py` implement:

```python
def normalize_answer(text: str) -> str: ...
def exact_agreement(a: str, b: str) -> bool: ...
def compute_iaa(records: list[dict]) -> dict: ...
def export_iaa_report(out_path: str, report: dict) -> None: ...
```

### Outputs

* `data/qa/qa_validation.jsonl`
* `data/qa/qa_iaa_subset.jsonl`
* `report_assets/intermediate/q1_dataset_stats.json`
* `report_assets/intermediate/q1_examples.json`

### Stop/go gate

Do not proceed until you have:

* at least 100 questions
* at least 30% double-annotated
* a computed IAA summary
* 3+ examples saved for the report

---

## Phase 3 — crawl and clean the EECS corpus

### Objective

Build your own retrieval corpus from EECS HTML pages.

### Required files

* `tools/crawl_eecs.py`
* `tools/clean_html_to_text.py`
* `tools/build_corpus.py`

### Required behavior

* stay inside `eecs.berkeley.edu`
* keep English HTML pages only
* exclude login-gated pages
* exclude PDFs and image-based evidence
* preserve titles, headings, paragraphs, useful table text
* remove scripts, menus, repeated boilerplate, footer noise
* deduplicate exact and near-duplicates

These constraints follow the assignment’s task scope and evidence rules. 

### Required functions

In `tools/crawl_eecs.py` implement:

```python
def should_visit(url: str) -> bool: ...
def crawl_site(start_url: str, out_dir: str, max_pages: int | None = None) -> None: ...
```

In `tools/clean_html_to_text.py` implement:

```python
def html_to_clean_text(html: str, url: str) -> dict: ...
def remove_boilerplate(text: str) -> str: ...
```

In `tools/build_corpus.py` implement:

```python
def deduplicate_pages(records: list[dict]) -> list[dict]: ...
def build_pages_jsonl(raw_html_dir: str, out_path: str) -> None: ...
def summarize_corpus(records: list[dict]) -> dict: ...
```

### Stop/go gate

Do not proceed until:

* `data/interim/pages.jsonl` exists
* every page record has URL, title, and cleaned text
* empty-page rate is low
* duplicates are removed
* a validation summary is written

---

## Phase 4 — chunk the corpus and build retrieval artifacts

### Objective

Prepare all runtime retrieval artifacts offline.

### Required files

* `tools/build_chunks.py`
* `tools/build_bm25_index.py`
* `tools/build_dense_index.py`

### Required functions

In `tools/build_chunks.py` implement:

```python
def chunk_text(text: str, target_words: int, overlap_words: int) -> list[str]: ...
def build_chunks(pages_path: str, out_path: str) -> None: ...
```

In `tools/build_bm25_index.py` implement:

```python
def tokenize_for_bm25(text: str) -> list[str]: ...
def build_bm25_artifacts(chunks_path: str, out_dir: str) -> None: ...
```

In `tools/build_dense_index.py` implement:

```python
def load_embedder(model_name: str): ...
def encode_chunks(chunks: list[dict], model_name: str) -> tuple[object, object]: ...
def build_faiss_index(embeddings): ...
def save_dense_artifacts(out_dir: str, embeddings, index, metadata: list[dict]) -> None: ...
```

### Outputs

* `data/interim/chunks.jsonl`
* `data/artifacts/bm25/...`
* `data/artifacts/dense/...`
* `data/artifacts/retrieval_manifest.json`

### Stop/go gate

Do not proceed until a fresh process can load both BM25 and FAISS artifacts successfully.

---

## Phase 5 — implement retrieval modules

### Objective

Create a debuggable hybrid retriever.

### Required files

* `src/retrieval/retrieval_types.py`
* `src/retrieval/bm25_retriever.py`
* `src/retrieval/dense_retriever.py`
* `src/retrieval/hybrid_retriever.py`

### Required dataclasses

In `retrieval_types.py` define:

```python
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    chunk_id: str
    page_id: str
    url: str
    title: str
    heading_path: str
    text: str
    score: float
    source: str

@dataclass
class RetrievalDebug:
    question: str
    retrieved: list[RetrievedChunk]
```

### Required interfaces

In `bm25_retriever.py`:

```python
class BM25Retriever:
    def __init__(self, artifact_dir: str): ...
    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]: ...
```

In `dense_retriever.py`:

```python
class DenseRetriever:
    def __init__(self, artifact_dir: str, model_name: str): ...
    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]: ...
```

In `hybrid_retriever.py`:

```python
class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever): ...
    def search(self, query: str, top_k: int = 6) -> list[RetrievedChunk]: ...
```

### Required behavior

* BM25 and dense retrieval must each work independently
* hybrid retrieval must fuse and deduplicate
* retrieval debug info must be savable during local evaluation

### Stop/go gate

Do not proceed until top-k retrieval on a manual sample shows sensible evidence URLs and answer-bearing chunks.

---

## Phase 6 — implement generation modules

### Objective

Generate short answers using only `llm.py`.

### Required files

* `src/generation/prompts.py`
* `src/generation/generator.py`
* `src/answer_postprocess.py`

### Required functions

In `prompts.py`:

```python
def build_qa_prompt(question: str, contexts: list[dict]) -> str: ...
def select_prompt_contexts(retrieved: list, max_contexts: int = 4) -> list[dict]: ...
```

In `generator.py`:

```python
class AnswerGenerator:
    def __init__(self, model_name: str, timeout_sec: int): ...
    def generate(self, question: str, contexts: list[dict]) -> str: ...
```

In `answer_postprocess.py`:

```python
def sanitize_answer(text: str) -> str: ...
def ensure_single_line(text: str) -> str: ...
def fallback_if_empty(text: str) -> str: ...
```

### Required prompt behavior

* answer only from provided context
* output `unknown` if unsupported
* no explanation
* under 10 words when possible
* exact `Yes` or `No` for yes/no questions

This aligns with the assignment’s answer format and evaluation behavior.

### Stop/go gate

Do not proceed until:

* missing `llm.py` fails cleanly
* generated outputs are single-line safe
* timeout failures return `unknown` instead of crashing

---

## Phase 7 — compose the full QA pipeline

### Objective

Create the end-to-end inference pipeline used by `run.sh`.

### Required file

* `src/pipeline/qa_pipeline.py`

### Required interface

```python
class QAPipeline:
    def __init__(self, config: dict): ...
    def answer_question(self, question: str) -> str: ...
    def answer_questions(self, questions: list[str]) -> list[str]: ...
```

### Required behavior

For each question:

1. retrieve candidates
2. select prompt contexts
3. generate answer
4. postprocess answer
5. optionally write debug traces

### Runtime rules

* initialize retrievers once
* initialize generator once
* load artifacts once
* no reloading per question
* logs go to stderr or debug files only

### Stop/go gate

Do not proceed until:

```bash
python3 -m src.main_predict --questions data/sample/questions.txt --output data/sample/predictions.txt
```

works without contaminating the output file.

---

## Phase 8 — implement evaluation, recall, and ablations

### Objective

Create the experimentation layer needed for model improvement and the report.

### Required files

* `src/eval/metrics.py`
* `src/eval/local_eval.py`
* `src/eval/recall_eval.py`
* `src/eval/ablations.py`
* `src/eval/error_analysis.py`

### Required functions

In `metrics.py`:

```python
def normalize_answer(text: str) -> str: ...
def exact_match(pred: str, golds: list[str]) -> float: ...
def token_f1(pred: str, golds: list[str]) -> float: ...
```

In `local_eval.py`:

```python
def run_local_eval(questions_path: str, predictions_out: str, results_out: str) -> None: ...
```

In `recall_eval.py`:

```python
def answer_string_recall(retrieved_texts: list[str], gold_answers: list[str]) -> float: ...
def url_match_recall(retrieved_urls: list[str], gold_url: str) -> float: ...
```

In `ablations.py`:

```python
def run_experiment(config: dict) -> dict: ...
def run_ablation_matrix(config_path: str) -> None: ...
```

In `error_analysis.py`:

```python
def categorize_error(record: dict) -> str: ...
def summarize_error_categories(records: list[dict]) -> dict: ...
```

### Required experiment outputs

Save:

* EM
* token F1
* retrieval recall
* latency
* timeout count
* experiment config
* notes

The report explicitly requires two ablations plus error analysis, and recommends looking at retrieval recall to understand whether errors come from retrieval or generation.

### Stop/go gate

Do not proceed until:

* local eval runs end-to-end
* at least one retrieval ablation is complete
* at least one corpus ablation is runnable after the reference corpus release

---

## Phase 9 — produce report-ready assets continuously

### Objective

Make report writing mostly an assembly task rather than a late scramble.

### Required outputs

* `report_assets/intermediate/q1_dataset_stats.json`
* `report_assets/intermediate/q1_examples.json`
* `report_assets/intermediate/q2_corpus_summary.json`
* `report_assets/intermediate/q3_system_summary.md`
* `report_assets/tables/ablation_results.csv`
* `report_assets/tables/error_categories.csv`
* `report_assets/intermediate/q6_takeaways.md`

### Required content mapping

* Q1: dataset creation, size, IAA, examples
* Q2: retrieval corpus construction and comparison with reference corpus
* Q3: full RAG architecture and design choices
* Q4: two most interesting ablations
* Q5: random subset of F1=0 errors with categories, including false negatives due to metric limitations
* Q6: takeaways and future ideas

---

## Phase 10 — submission validation and packaging

### Objective

Catch submission-killing mistakes before zip creation.

### Required files

* `tools/validate_submission.py`
* `tools/package_submission.py`

### Required validation checks

`validate_submission.py` must check:

* `run.sh` exists
* `run.sh` is executable
* `run.sh` accepts exactly two args
* required artifacts exist
* output line count matches input line count
* no output line contains embedded newline characters
* no absolute local paths are hardcoded
* no direct OpenRouter or forbidden API usage exists outside `llm.py`
* required QA data files are included

### Required packaging behavior

`package_submission.py` must:

* stage runtime code
* stage retrieval artifacts
* stage QA data created by the team
* exclude unnecessary scratch artifacts
* leave a clean `submission/` tree or zip-ready directory

The final submission must include the code/model artifact including retrieval datastore and the QA data you created. 

### Stop/go gate

Do not mark the repo complete until validation passes.

---

## 7. Minimum ablation matrix

Run and save at least these:

| ID | Retriever | Chunk Size | Prompt Contexts | Corpus    | LLM                              |
| -- | --------- | ---------: | --------------: | --------- | -------------------------------- |
| A1 | BM25      |        160 |               4 | own       | qwen-2.5-7b-instruct             |
| A2 | Dense     |        160 |               4 | own       | qwen-2.5-7b-instruct             |
| A3 | Hybrid    |        160 |               4 | own       | qwen-2.5-7b-instruct             |
| A4 | Hybrid    |        120 |               4 | own       | qwen-2.5-7b-instruct             |
| A5 | Hybrid    |        200 |               4 | own       | qwen-2.5-7b-instruct             |
| A6 | Hybrid    |        160 |               2 | own       | qwen-2.5-7b-instruct             |
| A7 | Hybrid    |        160 |               4 | reference | qwen-2.5-7b-instruct             |
| A8 | Hybrid    |        160 |               4 | own       | meta-llama/llama-3.1-8b-instruct |

---

## 8. Required commands the repo must support

```bash
python3 tools/crawl_eecs.py --start-url https://eecs.berkeley.edu --out data/raw/html
python3 tools/clean_html_to_text.py --in data/raw/html --out data/interim/pages.jsonl
python3 tools/build_chunks.py --in data/interim/pages.jsonl --out data/interim/chunks.jsonl
python3 tools/build_bm25_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/bm25
python3 tools/build_dense_index.py --in data/interim/chunks.jsonl --out-dir data/artifacts/dense
python3 tools/make_validation_set.py --out data/qa/qa_validation.jsonl
python3 tools/compute_iaa.py --gold data/qa/qa_validation.jsonl --subset data/qa/qa_iaa_subset.jsonl --out report_assets/intermediate/q1_dataset_stats.json
python3 -m src.eval.local_eval --questions data/qa/qa_validation.jsonl --predictions-out data/runs/predictions_dev.txt --results-out data/runs/experiment_results.csv
python3 -m src.eval.ablations --config config/experiment.yaml
python3 tools/validate_submission.py
bash run.sh data/sample/questions.txt data/sample/predictions.txt
```

---

## 9. Testing requirements

Create at least these tests or equivalent checks:

* `tests/test_io.py`
* `tests/test_postprocess.py`
* `tests/test_metrics.py`
* `tests/test_smoke_run.py`

The smoke test must verify that `run.sh` creates an output file with the same number of lines as the input file and that each line is newline-safe. That directly targets the autograder contract. 

---

## 10. README requirements

The README must explain:

* what the repo does
* blank-repo bootstrap assumption
* offline preprocessing flow
* local evaluation flow
* ablation flow
* final runtime flow
* expected artifact locations
* what happens if `llm.py` is missing
* how to validate before submission

---

## 11. Definition of done

The repository is complete only when all of these are true:

1. A blank directory can be turned into the final repo solely by files you created.
2. `bash run.sh <questions_txt_path> <predictions_out_path>` works with prebuilt artifacts. 
3. Runtime uses only allowed dependencies and stays inside CPU-only 4GB RAM limits. 
4. All LLM calls route only through `llm.py`.
5. The submission package contains code, retrieval artifacts/datastore, and the QA data created by the team. 
6. The repo contains enough outputs to answer all required report questions.
7. Submission validation passes locally.
8. No runtime step crawls the web or rebuilds indexes.
