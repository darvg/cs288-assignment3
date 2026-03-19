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
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += f" {text}"
        self.parts.append(text)


def remove_boilerplate(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    generic_patterns = [
        r"Skip to Content",
        r"Expand Main Menu",
        r"Collapse Main Menu",
        r"Expand Search Form",
        r"Collapse Search Form",
        r"Expand Submenu",
        r"Search for:\s*Search",
        r"For Students For Faculty/Staff Industry News Events Give",
        r"About About",
        r"Academics Academics",
        r"Research Research",
        r"People People",
        r"Connect Connect",
        r"Resources Resources",
        r"Blog Academics",
        r"EE CS UC Berkeley Berkeley Engineering CDSS",
        r"Accessibility Nondiscrimination Privacy Contact",
        r"View Open Faculty Positions",
        r"Learn more about the Campaign for Berkeley and Graduate Fellowships\. Give to EECS",
        r"Berkeley EECS on Twitter Berkeley EECS on Instagram Berkeley EECS on LinkedIn Berkeley EECS on YouTube",
        r"Cookie Policy",
        r"Privacy Policy",
    ]
    for pattern in generic_patterns:
        text = re.sub(pattern, " ", text, flags=re.I)

    breadcrumb = re.search(r"\bHome\s*/\s*[^ ]", text)
    if breadcrumb:
        text = text[breadcrumb.start() :]
        text = re.sub(r"^Home\s*/\s*", "", text)

    footer_markers = [
        r"\bAbout History Diversity Visiting Special Events\b",
        r"\bConsider reaching out for a conversation\b",
        r"\bContact us\b",
        r"©\s*\d{4}\s*UC Regents",
    ]
    for marker in footer_markers:
        match = re.search(marker, text, flags=re.I)
        if match:
            text = text[: match.start()]
            break

    text = re.sub(r"\b(skip to main content|cookie policy|privacy policy)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
