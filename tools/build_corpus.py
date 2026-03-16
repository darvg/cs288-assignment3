from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl, write_json, write_jsonl


def deduplicate_pages(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for record in records:
        digest = hashlib.md5(record["text"].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(record)
    return deduped


def build_pages_jsonl(raw_html_dir: str, out_path: str) -> None:
    from clean_html_to_text import html_to_clean_text

    records: list[dict] = []
    for html_path in sorted(Path(raw_html_dir).glob("*.html")):
        url_path = html_path.with_suffix(".url")
        url = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else html_path.name
        clean = html_to_clean_text(html_path.read_text(encoding="utf-8"), url)
        clean["page_id"] = f"page_{len(records):04d}"
        records.append(clean)
    records = deduplicate_pages(records)
    write_jsonl(out_path, records)
    summary = summarize_corpus(records)
    write_json("report_assets/intermediate/q2_corpus_summary.json", summary)


def summarize_corpus(records: list[dict]) -> dict:
    return {
        "num_pages": len(records),
        "avg_chars": round(sum(len(record["text"]) for record in records) / max(len(records), 1), 2),
        "urls": [record["url"] for record in records[:5]],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-html-dir", default="data/raw/html")
    parser.add_argument("--out", default="data/interim/pages.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_pages_jsonl(args.raw_html_dir, args.out)
