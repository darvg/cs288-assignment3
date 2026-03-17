from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def collect_upload_paths(
    pages_path: Path,
    chunks_path: Path | None,
    benchmark_path: Path | None,
) -> dict[str, Path]:
    uploads: dict[str, Path] = {"data/pages.jsonl": pages_path}
    if chunks_path is not None:
        uploads["data/chunks.jsonl"] = chunks_path
    if benchmark_path is not None:
        uploads["data/qa_live_benchmark.jsonl"] = benchmark_path
    return uploads


def build_metadata(
    pages_path: Path,
    chunks_path: Path | None,
    benchmark_path: Path | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "pages_path": str(pages_path),
        "num_pages": count_jsonl_records(pages_path),
    }
    if chunks_path is not None:
        metadata["chunks_path"] = str(chunks_path)
        metadata["num_chunks"] = count_jsonl_records(chunks_path)
    if benchmark_path is not None:
        metadata["benchmark_path"] = str(benchmark_path)
        metadata["num_benchmark_examples"] = count_jsonl_records(benchmark_path)
    return metadata


def build_dataset_card(metadata: dict[str, Any]) -> str:
    lines = [
        "# EECS Cleaned Corpus",
        "",
        "Cleaned UC Berkeley EECS crawl data exported from the CS288 assignment repository.",
        "",
        "## Contents",
        "",
        f"- `data/pages.jsonl`: {metadata['num_pages']} cleaned pages",
    ]
    if "num_chunks" in metadata:
        lines.append(f"- `data/chunks.jsonl`: {metadata['num_chunks']} retrieval chunks")
    if "num_benchmark_examples" in metadata:
        lines.append(
            f"- `data/qa_live_benchmark.jsonl`: {metadata['num_benchmark_examples']} benchmark examples"
        )
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "- `pages.jsonl`: `url`, `title`, `text`, `page_id`",
            "- `chunks.jsonl`: chunk-level retrieval records from the offline build pipeline",
            "",
            "This export is produced offline from the repository's cleaned corpus artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_optional_dependency(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise SystemExit(
            f"Missing optional dependency `{package_name}`. "
            f"Install it first, e.g. `pip install {package_name}`."
        ) from exc


def upload_files_mode(
    repo_id: str,
    token: str | None,
    private: bool,
    branch: str,
    uploads: dict[str, Path],
    metadata: dict[str, Any],
    commit_message: str,
) -> None:
    hub = ensure_optional_dependency("huggingface_hub", "huggingface_hub>=0.23")
    api = hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for path_in_repo, local_path in uploads.items():
        print(f"[upload] {local_path} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            commit_message=commit_message,
        )
    print("[upload] README.md")
    api.upload_file(
        path_or_fileobj=build_dataset_card(metadata).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        commit_message=commit_message,
    )
    print("[upload] metadata.json")
    api.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2).encode("utf-8"),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        commit_message=commit_message,
    )


def push_dataset_mode(
    repo_id: str,
    token: str | None,
    private: bool,
    split: str,
    branch: str,
    pages_path: Path,
    uploads: dict[str, Path],
    metadata: dict[str, Any],
    commit_message: str,
) -> None:
    hub = ensure_optional_dependency("huggingface_hub", "huggingface_hub>=0.23")
    datasets = ensure_optional_dependency("datasets", "datasets>=2.19")
    api = hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    records = read_jsonl(str(pages_path))
    print(f"[push] pages dataset split={split} rows={len(records)}")
    dataset = datasets.Dataset.from_list(records)
    dataset.push_to_hub(repo_id, split=split, token=token, private=private)
    extra_uploads = {k: v for k, v in uploads.items() if v != pages_path}
    for path_in_repo, local_path in extra_uploads.items():
        print(f"[upload] {local_path} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            commit_message=commit_message,
        )
    api.upload_file(
        path_or_fileobj=build_dataset_card(metadata).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        commit_message=commit_message,
    )
    api.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2).encode("utf-8"),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        commit_message=commit_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. username/eecs-cleaned")
    parser.add_argument("--pages", default="data/interim/pages.jsonl")
    parser.add_argument("--chunks", default="data/interim/chunks.jsonl")
    parser.add_argument("--benchmark", default="data/qa/qa_live_benchmark.jsonl")
    parser.add_argument("--mode", choices=["files", "dataset"], default="files")
    parser.add_argument("--split", default="train", help="Split name when using --mode dataset")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages_path = Path(args.pages)
    chunks_path = None if args.skip_chunks else Path(args.chunks)
    benchmark_path = None if args.skip_benchmark else Path(args.benchmark)

    if not pages_path.exists():
        raise SystemExit(f"Missing pages file: {pages_path}")
    if chunks_path is not None and not chunks_path.exists():
        raise SystemExit(f"Missing chunks file: {chunks_path}")
    if benchmark_path is not None and not benchmark_path.exists():
        raise SystemExit(f"Missing benchmark file: {benchmark_path}")

    uploads = collect_upload_paths(pages_path, chunks_path, benchmark_path)
    metadata = build_metadata(pages_path, chunks_path, benchmark_path)
    token = os.environ.get(args.token_env)
    commit_message = "Upload cleaned EECS corpus artifacts"

    print(f"[repo] {args.repo_id}")
    print(f"[mode] {args.mode}")
    print(f"[metadata] {json.dumps(metadata, indent=2)}")
    for path_in_repo, local_path in uploads.items():
        print(f"[plan] {local_path} -> {path_in_repo}")

    if args.dry_run:
        return

    if args.mode == "files":
        upload_files_mode(
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            branch=args.branch,
            uploads=uploads,
            metadata=metadata,
            commit_message=commit_message,
        )
    else:
        push_dataset_mode(
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            split=args.split,
            branch=args.branch,
            pages_path=pages_path,
            uploads=uploads,
            metadata=metadata,
            commit_message=commit_message,
        )


if __name__ == "__main__":
    main()
