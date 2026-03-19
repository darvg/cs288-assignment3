from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.text_utils import simple_tokenize


HASH_DIM = 1024


def load_embedder(model_name: str):
    return {"model_name": model_name}


def encode_chunks(chunks: list[dict], model_name: str) -> tuple[object, object]:
    _ = load_embedder(model_name)
    embeddings = np.zeros((len(chunks), HASH_DIM), dtype=np.float32)
    for row, chunk in enumerate(chunks):
        for token in simple_tokenize(chunk["text"]):
            embeddings[row, _hash_index(token)] += 1.0
        norm = np.linalg.norm(embeddings[row])
        if norm:
            embeddings[row] /= norm
    return embeddings, {"hash_dim": HASH_DIM}


def build_faiss_index(embeddings):
    return {"type": "IndexFlatIP", "shape": list(embeddings.shape)}


def save_dense_artifacts(out_dir: str, embeddings, index, metadata: list[dict]) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "embeddings.npy", embeddings.astype(np.float16))
    write_jsonl(str(target / "metadata.jsonl"), metadata)
    write_json(
        str(target / "index.json"),
        {
            "type": index["type"],
            "shape": index["shape"],
            "hash_dim": index["hash_dim"],
            "dtype": "float16",
        },
    )


def build_dense(chunks_path: str, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    chunks = read_jsonl(chunks_path)
    embeddings, encoder_info = encode_chunks(chunks, model_name)
    index = build_faiss_index(embeddings)
    index["hash_dim"] = encoder_info["hash_dim"]
    save_dense_artifacts(out_dir, embeddings, index, chunks)
    write_json(
        "data/artifacts/retrieval_manifest.json",
        {
            "dense_model": model_name,
            "num_chunks": len(chunks),
            "dense_dir": out_dir,
            "hash_dim": encoder_info["hash_dim"],
        },
    )


def _hash_index(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % HASH_DIM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dense(args.input_path, args.out_dir)
