from __future__ import annotations

from .retrieval_types import RetrievedChunk


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    def search(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        bm25 = self.bm25_retriever.search(query, top_k=max(top_k, 10))
        dense = self.dense_retriever.search(query, top_k=max(top_k, 10))
        combined: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(bm25, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
            combined[chunk.chunk_id] = chunk
        for rank, chunk in enumerate(dense, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
            if chunk.chunk_id not in combined:
                combined[chunk.chunk_id] = chunk

        ranked = sorted(
            combined.values(),
            key=lambda chunk: scores.get(chunk.chunk_id, 0.0),
            reverse=True,
        )
        fused: list[RetrievedChunk] = []
        for chunk in ranked[:top_k]:
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    page_id=chunk.page_id,
                    url=chunk.url,
                    title=chunk.title,
                    heading_path=chunk.heading_path,
                    text=chunk.text,
                    score=scores[chunk.chunk_id],
                    source="hybrid",
                )
            )
        return fused
