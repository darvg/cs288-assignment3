from __future__ import annotations

import math
from pathlib import Path

from ..io_utils import read_json, read_jsonl
from ..text_utils import simple_tokenize
from .retrieval_types import RetrievedChunk


class BM25Retriever:
    def __init__(self, artifact_dir: str):
        artifact_root = Path(artifact_dir)
        self.metadata = read_jsonl(str(artifact_root / "metadata.jsonl"))
        self.doc_freq = read_json(str(artifact_root / "doc_freq.json"))
        self.doc_lengths = read_json(str(artifact_root / "doc_lengths.json"))
        self.avg_doc_len = float(read_json(str(artifact_root / "stats.json"))["avg_doc_len"])
        self.num_docs = len(self.metadata)
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        tokens = simple_tokenize(query)
        scored: list[RetrievedChunk] = []
        for idx, record in enumerate(self.metadata):
            tf = record.get("token_counts", {})
            dl = self.doc_lengths[str(idx)]
            score = 0.0
            for token in tokens:
                df = self.doc_freq.get(token, 0)
                if not df:
                    continue
                freq = tf.get(token, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
                score += idf * ((freq * (self.k1 + 1)) / denom)
            if score > 0:
                scored.append(_to_chunk(record, score, "bm25"))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


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
