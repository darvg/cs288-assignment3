from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def should_visit(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.endswith("eecs.berkeley.edu")
        and not parsed.path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif"))
    )


def crawl_site(start_url: str, out_dir: str, max_pages: int | None = None) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    queue = [start_url]
    while queue and (max_pages is None or len(seen) < max_pages):
        url = queue.pop(0)
        if url in seen or not should_visit(url):
            continue
        seen.add(url)
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "")
                if "html" not in content_type:
                    continue
                html = response.read().decode("utf-8", errors="ignore")
        except urllib.error.URLError:
            continue
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        (out_path / f"{digest}.html").write_text(html, encoding="utf-8")
        (out_path / f"{digest}.url").write_text(url, encoding="utf-8")
        parser = _LinkParser()
        parser.feed(html)
        for href in parser.links:
            next_url = urllib.parse.urljoin(url, href)
            if next_url not in seen and should_visit(next_url):
                queue.append(next_url)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pages", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    crawl_site(args.start_url, args.out, args.max_pages)
