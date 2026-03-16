from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.text_utils import simple_tokenize


def tokenize_for_bm25(text: str) -> list[str]:
    return simple_tokenize(text)


def build_bm25_artifacts(chunks_path: str, out_dir: str) -> None:
    chunks = read_jsonl(chunks_path)
    doc_freq: Counter[str] = Counter()
    doc_lengths: dict[str, int] = {}
    metadata: list[dict] = []
    for index, chunk in enumerate(chunks):
        tokens = tokenize_for_bm25(chunk["text"])
        counts = Counter(tokens)
        for token in counts:
            doc_freq[token] += 1
        doc_lengths[str(index)] = len(tokens)
        enriched = dict(chunk)
        enriched["token_counts"] = dict(counts)
        metadata.append(enriched)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(target / "metadata.jsonl"), metadata)
    write_json(str(target / "doc_freq.json"), dict(doc_freq))
    write_json(str(target / "doc_lengths.json"), doc_lengths)
    avg_doc_len = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
    write_json(str(target / "stats.json"), {"avg_doc_len": avg_doc_len, "num_docs": len(doc_lengths)})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_bm25_artifacts(args.input_path, args.out_dir)
