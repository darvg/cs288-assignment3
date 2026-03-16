from __future__ import annotations

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
