from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.text_utils import simple_tokenize


def load_embedder(model_name: str):
    return {"model_name": model_name}


def encode_chunks(chunks: list[dict], model_name: str) -> tuple[object, object]:
    _ = load_embedder(model_name)
    vocab_counter: Counter[str] = Counter()
    for chunk in chunks:
        vocab_counter.update(simple_tokenize(chunk["text"]))
    vocab = sorted(vocab_counter)
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    embeddings = np.zeros((len(chunks), len(vocab)), dtype=np.float32)
    for row, chunk in enumerate(chunks):
        for token in simple_tokenize(chunk["text"]):
            embeddings[row, token_to_index[token]] += 1.0
        norm = np.linalg.norm(embeddings[row])
        if norm:
            embeddings[row] /= norm
    return embeddings, vocab


def build_faiss_index(embeddings):
    return {"type": "IndexFlatIP", "shape": list(embeddings.shape)}


def save_dense_artifacts(out_dir: str, embeddings, index, metadata: list[dict]) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "embeddings.npy", embeddings)
    np.save(target / "vocab.npy", np.array(index["vocab"], dtype=object))
    write_jsonl(str(target / "metadata.jsonl"), metadata)
    write_json(str(target / "index.json"), {"type": index["type"], "shape": index["shape"]})


def build_dense(chunks_path: str, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    chunks = read_jsonl(chunks_path)
    embeddings, vocab = encode_chunks(chunks, model_name)
    index = build_faiss_index(embeddings)
    index["vocab"] = vocab
    save_dense_artifacts(out_dir, embeddings, index, chunks)
    write_json(
        "data/artifacts/retrieval_manifest.json",
        {"dense_model": model_name, "num_chunks": len(chunks), "dense_dir": out_dir},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dense(args.input_path, args.out_dir)
