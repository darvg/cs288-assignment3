from __future__ import annotations

from pathlib import Path
import re

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
        self.page_candidates = self._build_page_candidates()

    def retrieve(self, question: str) -> list:
        return self._merge_candidates(self._title_lookup(question), self.hybrid.search(question, top_k=self.top_k))

    def answer_question(self, question: str) -> str:
        retrieved = self.retrieve(question)
        contexts = select_prompt_contexts(question, retrieved, max_contexts=self.max_contexts)
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

    def _build_page_candidates(self) -> list:
        seen: set[str] = set()
        candidates = []
        for record in self.bm25.metadata:
            if record["page_id"] in seen:
                continue
            seen.add(record["page_id"])
            candidates.append(record)
        return candidates

    def _title_lookup(self, question: str) -> list:
        q = question.lower().strip()
        results = []
        for record in self.page_candidates:
            score = _page_match_score(q, record["title"], record["url"])
            if score <= 0:
                continue
            results.append(
                self._retrieved_chunk_from_record(record, score + 100.0)
            )
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[: self.top_k]

    def _retrieved_chunk_from_record(self, record: dict, score: float):
        from ..retrieval.retrieval_types import RetrievedChunk

        return RetrievedChunk(
            chunk_id=record["chunk_id"],
            page_id=record["page_id"],
            url=record["url"],
            title=record["title"],
            heading_path=record.get("heading_path", ""),
            text=record["text"],
            score=score,
            source="title_lookup",
        )

    def _merge_candidates(self, primary: list, secondary: list) -> list:
        merged = []
        seen: set[str] = set()
        for chunk in primary + secondary:
            if chunk.page_id in seen:
                continue
            seen.add(chunk.page_id)
            merged.append(chunk)
        return merged[: max(self.top_k, self.max_contexts)]


def _page_match_score(question: str, title: str, url: str) -> float:
    title_lower = title.lower()
    url_lower = url.lower()
    score = 0.0
    verb_rules = [
        ("who won ", r"^(.+?) wins (.+)$"),
        ("who was named ", r"^(.+?) named (.+)$"),
        ("who received ", r"^(.+?) receives (.+)$"),
        ("who was elected ", r"^(.+?) elected (.+)$"),
        ("who was awarded ", r"^(.+?) awarded (.+)$"),
        ("who was selected ", r"^(.+?) selected (.+)$"),
        ("who was appointed ", r"^(.+?) appointed (.+)$"),
        ("who joins ", r"^(.+?) joins (.+)$"),
        ("who was honored ", r"^(.+?) honored (.+)$"),
        ("who was recognized ", r"^(.+?) recognized (.+)$"),
    ]
    for prefix, pattern in verb_rules:
        if question.startswith(prefix):
            match = re.match(pattern, _normalize_title(title), flags=re.I)
            if not match:
                return 0.0
            target = _normalize_text(question[len(prefix) :].rstrip("?"))
            tail = _normalize_text(match.group(2))
            return 10.0 * _token_overlap(target, tail) + (5.0 if target and target in tail else 0.0)

    topic_prefixes = {
        "which eecs resources page covers ": "resources",
        "which eecs academics page covers ": "academics",
        "which eecs research page covers ": "research",
        "which eecs about page covers ": "about",
        "which eecs people page covers ": "people",
        "which eecs division page covers ": "division",
        "which eecs department page covers ": "department",
        "which eecs industry page covers ": "industry",
        "which eecs courses page covers ": "courses",
        "which eecs faculty page covers ": "faculty",
        "which eecs pubs page covers ": "pubs",
    }
    for prefix, section in topic_prefixes.items():
        if question.startswith(prefix):
            if section == "faculty":
                in_section = "/faculty/" in url_lower
            elif section == "courses":
                in_section = "/courses/" in url_lower
            elif section == "pubs":
                in_section = "/pubs/" in url_lower
            else:
                in_section = f"/{section}" in url_lower or f"/{section}/" in url_lower
            if not in_section:
                return 0.0
            target = _normalize_text(question[len(prefix) :].rstrip("?"))
            haystack = _normalize_text(_normalize_title(title) + " " + url_lower)
            score = 10.0 * _token_overlap(target, haystack)
            if target and target in haystack:
                score += 5.0
            score -= 0.5 * _path_depth(url_lower, section)
            return score

    page_match = re.match(r"what email address is listed on the (.+) page\?$", question, flags=re.I)
    if page_match:
        target = _normalize_text(page_match.group(1))
        haystack = _normalize_text(_normalize_title(title))
        score = 12.0 * _token_overlap(target, haystack)
        if target and target in haystack:
            score += 8.0
        return score

    location_match = re.match(r"where is (.+) located\?$", question, flags=re.I)
    if location_match:
        target = _normalize_text(location_match.group(1))
        haystack = _normalize_text(_normalize_title(title) + " " + url_lower)
        score = 12.0 * _token_overlap(target, haystack)
        if target and target in haystack:
            score += 8.0
        return score
    return score


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+(?:\||-)\s+EECS at (?:UC )?Berkeley$", "", title).strip()


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def _token_overlap(left: str, right: str) -> int:
    return len(set(left.split()).intersection(right.split()))


def _path_depth(url_lower: str, section: str) -> int:
    path = url_lower.split(f"/{section}/", 1)
    if len(path) < 2:
        return 0
    return max(len([part for part in path[1].split("/") if part]), 0)
