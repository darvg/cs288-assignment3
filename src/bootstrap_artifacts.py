from __future__ import annotations

import os
from pathlib import Path
import urllib.request


RUNTIME_ARTIFACT_PATHS = [
    "data/artifacts/retrieval_manifest.json",
    "data/interim/chunks.jsonl",
    "data/artifacts/bm25/doc_freq.json",
    "data/artifacts/bm25/doc_lengths.json",
    "data/artifacts/bm25/metadata.jsonl",
    "data/artifacts/bm25/stats.json",
    "data/artifacts/dense/embeddings.npy",
    "data/artifacts/dense/index.json",
    "data/artifacts/dense/metadata.jsonl",
    "data/artifacts/dense/vocab.npy",
]


def ensure_runtime_artifacts(config: dict) -> None:
    root = Path(config["resolved_project_root"])
    missing = [root / rel for rel in RUNTIME_ARTIFACT_PATHS if not (root / rel).exists()]
    if not missing:
        return

    bootstrap = config.get("bootstrap", {})
    if bootstrap and bootstrap.get("enabled", True) is False:
        return

    base_url = _bootstrap_base_url(config)
    if not base_url:
        missing_list = ", ".join(str(path.relative_to(root)) for path in missing[:4])
        raise FileNotFoundError(
            "Runtime corpus artifacts are missing. "
            f"Set RAG_CORPUS_BASE_URL (or bootstrap.base_url) to auto-download them. "
            f"Missing examples: {missing_list}"
        )

    for path in missing:
        rel_path = path.relative_to(root).as_posix()
        download_url = f"{base_url.rstrip('/')}/{rel_path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[bootstrap] downloading {download_url}", flush=True)
        with urllib.request.urlopen(download_url, timeout=60) as response:
            path.write_bytes(response.read())


def _bootstrap_base_url(config: dict) -> str:
    bootstrap = config.get("bootstrap", {})
    env_name = bootstrap.get("base_url_env", "RAG_CORPUS_BASE_URL")
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    return str(bootstrap.get("base_url", "")).strip()
