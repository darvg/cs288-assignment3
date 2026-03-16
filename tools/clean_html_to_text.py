from __future__ import annotations

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import write_jsonl


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += f" {text}"
        self.parts.append(text)


def remove_boilerplate(text: str) -> str:
    text = re.sub(r"\b(skip to main content|cookie policy|privacy policy)\b", " ", text, flags=re.I)
    return " ".join(text.split())


def html_to_clean_text(html_text: str, url: str) -> dict:
    parser = _TextParser()
    parser.feed(html_text)
    text = remove_boilerplate(html.unescape(" ".join(parser.parts)))
    title = " ".join(parser.title.split()) or url
    return {"url": url, "title": title, "text": text}


def _run(in_dir: str, out_path: str) -> None:
    records: list[dict] = []
    for html_path in sorted(Path(in_dir).glob("*.html")):
        url_path = html_path.with_suffix(".url")
        url = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else html_path.name
        records.append(html_to_clean_text(html_path.read_text(encoding="utf-8"), url))
    write_jsonl(out_path, records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run(args.in_dir, args.out)
