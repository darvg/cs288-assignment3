from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io_utils import read_jsonl
from ..text_utils import simple_tokenize
from .retrieval_types import RetrievedChunk


class DenseRetriever:
    def __init__(self, artifact_dir: str, model_name: str):
        artifact_root = Path(artifact_dir)
        self.model_name = model_name
        self.metadata = read_jsonl(str(artifact_root / "metadata.jsonl"))
        self.embeddings = np.load(artifact_root / "embeddings.npy")
        self.vocab = {}
        for index, token in enumerate(np.load(artifact_root / "vocab.npy", allow_pickle=True).tolist()):
            self.vocab[token] = index

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query_vector = self._encode(query)
        scores = self.embeddings @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        return [_to_chunk(self.metadata[i], float(scores[i]), "dense") for i in order if scores[i] > 0]

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        for token in simple_tokenize(text):
            index = self.vocab.get(token)
            if index is not None:
                vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector


def _to_chunk(record: dict, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=record["chunk_id"],
        page_id=record["page_id"],
        url=record["url"],
        title=record["title"],
        heading_path=record.get("heading_path", ""),
        text=record["text"],
        score=float(score),
        source=source,
    )
