from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_jsonl
from src.text_utils import split_words


def chunk_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    words = split_words(text)
    if not words:
        return []
    step = max(target_words - overlap_words, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + target_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + target_words >= len(words):
            break
    return chunks


def build_chunks(pages_path: str, out_path: str) -> None:
    pages = read_jsonl(pages_path)
    chunks: list[dict] = []
    for page in pages:
        for index, chunk in enumerate(chunk_text(page["text"], 160, 40)):
            chunks.append(
                {
                    "chunk_id": f"{page['page_id']}_chunk_{index:03d}",
                    "page_id": page["page_id"],
                    "url": page["url"],
                    "title": page["title"],
                    "heading_path": page.get("title", ""),
                    "text": chunk,
                }
            )
    write_jsonl(out_path, chunks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_chunks(args.input_path, args.out)
