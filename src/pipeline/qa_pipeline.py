from __future__ import annotations

from pathlib import Path

from ..answer_postprocess import ensure_single_line, fallback_if_empty
from ..config_utils import resolve_project_root
from ..io_utils import ensure_parent_dir, write_json
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generation.generator import AnswerGenerator
from ..generation.prompts import select_prompt_contexts


class QAPipeline:
    def __init__(self, config: dict):
        self.config = config
        root = Path(config.get("resolved_project_root", resolve_project_root()))
        artifacts = config["artifacts"]
        self.bm25 = BM25Retriever(str(root / artifacts["bm25_dir"]))
        self.dense = DenseRetriever(str(root / artifacts["dense_dir"]), config["runtime"]["model_name"])
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.generator = AnswerGenerator(
            model_name=config["runtime"]["model_name"],
            timeout_sec=int(config["runtime"]["timeout_sec"]),
        )
        self.top_k = int(config["runtime"]["top_k"])
        self.max_contexts = int(config["runtime"]["max_contexts"])
        self.use_llm = bool(config["runtime"]["use_llm"])
        self.debug_enabled = bool(config.get("debug", {}).get("enabled", False))
        self.debug_dir = root / config.get("debug", {}).get("output_dir", "data/runs/debug")

    def answer_question(self, question: str) -> str:
        retrieved = self.hybrid.search(question, top_k=self.top_k)
        contexts = select_prompt_contexts(retrieved, max_contexts=self.max_contexts)
        answer = self.generator.generate(question, contexts, use_llm=self.use_llm)
        answer = ensure_single_line(fallback_if_empty(answer))
        if self.debug_enabled:
            self._write_debug(question, retrieved, answer)
        return answer

    def answer_questions(self, questions: list[str]) -> list[str]:
        return [self.answer_question(question) for question in questions]

    def _write_debug(self, question: str, retrieved: list, answer: str) -> None:
        ensure_parent_dir(str(self.debug_dir / "placeholder.json"))
        payload = {
            "question": question,
            "answer": answer,
            "retrieved": [chunk.__dict__ for chunk in retrieved],
        }
        filename = f"debug_{abs(hash(question))}.json"
        write_json(str(self.debug_dir / filename), payload)
