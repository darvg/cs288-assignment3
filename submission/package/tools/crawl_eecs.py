from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - offline crawler only
    BeautifulSoup = None

DEFAULT_SITEMAP_INDEX = "https://eecs.berkeley.edu/sitemap_index.xml"
DEFAULT_CRAWL_DELAY_SEC = 1.3
USER_AGENT = "cs288-a3-rag-crawler/1.0"
ALLOWED_HOST_RE = re.compile(r"^(?:www\d*\.)?eecs\.berkeley\.edu$", re.I)


def canonicalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = urllib.parse.quote(urllib.parse.unquote(parsed.path or "/"), safe="/:@-._~!$&'()*+,;=")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    filtered_query = [(k, v) for k, v in query if not k.lower().startswith("utm_")]
    return urllib.parse.urlunparse(
        (scheme, netloc, path, "", urllib.parse.urlencode(filtered_query), "")
    )


def should_visit(url: str) -> bool:
    if not url or any(character in url for character in "\r\n\t"):
        return False
    if len(url) > 2048:
        return False
    if url.count(" ") > 3:
        return False
    parsed = urllib.parse.urlparse(canonicalize_url(url))
    lower_path = parsed.path.lower()
    disallowed_suffixes = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".zip",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".mp4",
        ".mp3",
    )
    if parsed.scheme not in {"http", "https"}:
        return False
    if not is_allowed_host(parsed.netloc):
        return False
    if lower_path.startswith("/pubs/techrpts/"):
        return False
    if lower_path.startswith("/book/") and len(parsed.path) > 300:
        return False
    if lower_path.startswith("/Pubs/"):
        return False
    if lower_path.endswith(disallowed_suffixes):
        return False
    return True


def effective_crawl_delay(start_url: str, requested_delay_sec: float) -> float:
    parsed = urllib.parse.urlparse(canonicalize_url(start_url))
    if is_allowed_host(parsed.netloc):
        return max(requested_delay_sec, DEFAULT_CRAWL_DELAY_SEC)
    return requested_delay_sec


def is_allowed_host(netloc: str) -> bool:
    host = netloc.lower().strip()
    if ":" in host:
        host = host.split(":", 1)[0]
    return bool(ALLOWED_HOST_RE.fullmatch(host))


def url_section(url: str) -> str:
    parsed = urllib.parse.urlparse(canonicalize_url(url))
    parts = [part for part in parsed.path.split("/") if part]
    return parts[0] if parts else "__root__"


class CrawlState:
    def __init__(self, out_dir: str, crawl_delay_sec: float) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "crawl_manifest.json"
        self.errors_path = self.out_dir / "crawl_errors.jsonl"
        self.frontier_path = self.out_dir / "crawl_frontier.json"
        self.crawl_delay_sec = crawl_delay_sec
        self.last_request_time = 0.0
        self.manifest = self._load_manifest()
        self.seen_urls = set(self.manifest["saved"].keys()) | set(self.manifest["failed"].keys())

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "saved": {},
            "failed": {},
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "crawl_delay_sec": self.crawl_delay_sec,
        }

    def wait_for_delay(self) -> None:
        remaining = self.crawl_delay_sec - (time.time() - self.last_request_time)
        if remaining > 0:
            print(f"[sleep] waiting {remaining:.1f}s for crawl delay", flush=True)
            time.sleep(remaining)

    def record_request(self) -> None:
        self.last_request_time = time.time()

    def save_success(self, url: str, html: str, content_type: str, status: int) -> None:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        (self.out_dir / f"{digest}.html").write_text(html, encoding="utf-8")
        (self.out_dir / f"{digest}.url").write_text(url, encoding="utf-8")
        metadata = {
            "url": url,
            "content_type": content_type,
            "status": status,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.out_dir / f"{digest}.meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.manifest["saved"][url] = metadata
        self.seen_urls.add(url)
        self._flush_manifest()
        print(f"[saved {len(self.manifest['saved'])}] {url}", flush=True)

    def save_failure(self, url: str, error: str) -> None:
        payload = {
            "url": url,
            "error": error,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.errors_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.manifest["failed"][url] = payload
        self.seen_urls.add(url)
        self._flush_manifest()
        print(f"[failed {len(self.manifest['failed'])}] {url} :: {error}", flush=True)

    def _flush_manifest(self) -> None:
        summary = {
            **self.manifest,
            "num_saved": len(self.manifest["saved"]),
            "num_failed": len(self.manifest["failed"]),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def print_summary(self, queue_size: int) -> None:
        print(
            f"[summary] saved={len(self.manifest['saved'])} "
            f"failed={len(self.manifest['failed'])} queue={queue_size}",
            flush=True,
        )

    def load_frontier(self) -> deque[str]:
        if not self.frontier_path.exists():
            return deque()
        payload = json.loads(self.frontier_path.read_text(encoding="utf-8"))
        pending = [url for url in payload.get("pending", []) if url not in self.seen_urls and should_visit(url)]
        if pending:
            print(f"[resume] restored {len(pending)} pending URLs from prior crawl state", flush=True)
        return deque(pending)

    def save_frontier(self, queue: deque[str]) -> None:
        pending: list[str] = []
        queued_seen: set[str] = set()
        for url in queue:
            if url in self.seen_urls or url in queued_seen or not should_visit(url):
                continue
            queued_seen.add(url)
            pending.append(url)
        payload = {
            "pending": pending,
            "num_pending": len(pending),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.frontier_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def saved_html_files(self) -> list[Path]:
        return sorted(self.out_dir.glob("*.html"))


def _fetch_url(url: str, state: CrawlState, timeout: int = 30) -> tuple[str, str, int]:
    state.wait_for_delay()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            state.record_request()
            content_type = response.headers.get("Content-Type", "")
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", errors="ignore")
            return body, content_type, status
    except (urllib.error.URLError, http.client.InvalidURL, ValueError) as exc:
        state.record_request()
        raise RuntimeError(str(exc)) from exc


def _parse_xml_locs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    locs: list[str] = []
    for element in root.iter():
        if element.tag.endswith("loc") and element.text:
            locs.append(element.text.strip())
    return locs


def _discover_sitemap_urls(sitemap_index_urls: list[str], state: CrawlState) -> list[str]:
    discovered: list[str] = []
    seen_indexes: set[str] = set()
    for sitemap_index_url in sitemap_index_urls:
        sitemap_index_url = canonicalize_url(sitemap_index_url)
        if sitemap_index_url in seen_indexes:
            continue
        seen_indexes.add(sitemap_index_url)
        print(f"[sitemap] fetching index {sitemap_index_url}", flush=True)
        try:
            xml_text, _, _ = _fetch_url(sitemap_index_url, state)
        except RuntimeError as exc:
            state.save_failure(sitemap_index_url, f"sitemap index fetch failed: {exc}")
            continue
        sitemap_urls = _parse_xml_locs(xml_text)
        print(f"[sitemap] discovered {len(sitemap_urls)} sitemap files from {sitemap_index_url}", flush=True)
        for sitemap_url in sitemap_urls:
            print(f"[sitemap] fetching sitemap {sitemap_url}", flush=True)
            try:
                sitemap_xml, _, _ = _fetch_url(sitemap_url, state)
            except RuntimeError as exc:
                state.save_failure(sitemap_url, f"sitemap fetch failed: {exc}")
                continue
            for url in _parse_xml_locs(sitemap_xml):
                if should_visit(url):
                    discovered.append(canonicalize_url(url))
    print(f"[sitemap] discovered {len(discovered)} candidate HTML URLs", flush=True)
    return discovered


def _diversify_urls(urls: list[str]) -> list[str]:
    buckets: dict[str, deque[str]] = defaultdict(deque)
    seen: set[str] = set()
    for url in urls:
        canonical = canonicalize_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        buckets[url_section(canonical)].append(canonical)
    ordered_sections = sorted(buckets.keys(), key=lambda section: (-len(buckets[section]), section))
    diversified: list[str] = []
    while True:
        progressed = False
        for section in ordered_sections:
            if buckets[section]:
                diversified.append(buckets[section].popleft())
                progressed = True
        if not progressed:
            break
    return diversified


def _extract_links(base_url: str, html: str) -> list[str]:
    if BeautifulSoup is None:
        hrefs = re.findall(r"""href=["']([^"'#]+)["']""", html, flags=re.I)
        links: list[str] = []
        for href in hrefs:
            if len(href) > 1024 or href.count(" ") > 3:
                continue
            candidate = canonicalize_url(urllib.parse.urljoin(base_url, href))
            if should_visit(candidate):
                links.append(candidate)
        return links

    soup = BeautifulSoup(html, "html.parser")
    canonical_link = soup.find("link", rel=lambda value: value and "canonical" in str(value).lower())
    if canonical_link and canonical_link.get("href"):
        canonical_candidate = canonicalize_url(urllib.parse.urljoin(base_url, canonical_link["href"]))
        base_url = canonical_candidate

    meta_robots = soup.find("meta", attrs={"name": lambda value: value and value.lower() == "robots"})
    if meta_robots and meta_robots.get("content"):
        directives = meta_robots["content"].lower()
        if "nofollow" in directives:
            return []

    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if not href:
            continue
        if len(href) > 1024 or href.count(" ") > 3:
            continue
        candidate = canonicalize_url(urllib.parse.urljoin(base_url, href))
        if should_visit(candidate):
            links.append(candidate)
    return links


def _reconstruct_frontier_from_saved_pages(state: CrawlState) -> deque[str]:
    recovered: list[str] = []
    queued_seen: set[str] = set()
    html_files = state.saved_html_files()
    print(f"[resume] rebuilding frontier from {len(html_files)} saved HTML pages", flush=True)
    for html_path in html_files:
        url_path = html_path.with_suffix(".url")
        if not url_path.exists():
            continue
        base_url = url_path.read_text(encoding="utf-8").strip()
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        for linked_url in _diversify_urls(_extract_links(base_url, html)):
            if linked_url in state.seen_urls or linked_url in queued_seen:
                continue
            queued_seen.add(linked_url)
            recovered.append(linked_url)
    if recovered:
        print(f"[resume] recovered {len(recovered)} pending URLs from saved pages", flush=True)
    return deque(recovered)


def crawl_site(
    start_urls: list[str],
    out_dir: str,
    max_pages: int | None = None,
    sitemap_index_urls: list[str] | None = None,
    crawl_delay_sec: float = DEFAULT_CRAWL_DELAY_SEC,
) -> None:
    canonical_start_urls = [canonicalize_url(url) for url in start_urls if should_visit(url)]
    if not canonical_start_urls:
        raise ValueError("No valid start URLs provided")
    sitemap_index_urls = _resolve_sitemap_index_urls(canonical_start_urls, sitemap_index_urls or [])
    state = CrawlState(out_dir=out_dir, crawl_delay_sec=effective_crawl_delay(canonical_start_urls[0], crawl_delay_sec))
    print(
        f"[start] start_urls={canonical_start_urls} out_dir={out_dir} "
        f"max_pages={max_pages} crawl_delay_sec={state.crawl_delay_sec} "
        f"sitemap_index_urls={sitemap_index_urls}",
        flush=True,
    )
    queue = state.load_frontier()
    if not queue:
        if state.manifest["saved"] or state.manifest["failed"]:
            queue = _reconstruct_frontier_from_saved_pages(state)
            if not queue:
                for start_url in canonical_start_urls:
                    queue.append(start_url)
                sitemap_urls = _diversify_urls(_discover_sitemap_urls(sitemap_index_urls, state))
                state.manifest["discovered_from_sitemaps"] = len(sitemap_urls)
                state._flush_manifest()
                print(
                    f"[queue] frontier file missing; reconstructed pending queue from sitemap minus "
                    f"{len(state.seen_urls)} seen URLs",
                    flush=True,
                )
                for url in sitemap_urls:
                    if url not in state.seen_urls:
                        queue.append(url)
            else:
                print(f"[queue] reconstructed {len(queue)} pending URLs from saved pages", flush=True)
        else:
            for start_url in canonical_start_urls:
                queue.append(start_url)
            sitemap_urls = _diversify_urls(_discover_sitemap_urls(sitemap_index_urls, state))
            state.manifest["discovered_from_sitemaps"] = len(sitemap_urls)
            state._flush_manifest()
            print(f"[queue] seeded {len(sitemap_urls)} diversified sitemap URLs", flush=True)
            for url in sitemap_urls:
                if url not in state.seen_urls:
                    queue.append(url)
        state.save_frontier(queue)
    else:
        print(f"[queue] resuming with {len(queue)} pending URLs", flush=True)

    while queue and (max_pages is None or len(state.manifest["saved"]) < max_pages):
        url = canonicalize_url(queue.popleft())
        state.save_frontier(queue)
        if url in state.seen_urls or not should_visit(url):
            continue
        print(f"[fetch] queue={len(queue)} url={url}", flush=True)
        try:
            html, content_type, status = _fetch_url(url, state)
        except RuntimeError as exc:
            state.save_failure(url, str(exc))
            state.print_summary(len(queue))
            continue
        if "html" not in content_type.lower():
            state.save_failure(url, f"skipped non-html content-type: {content_type}")
            state.print_summary(len(queue))
            continue
        state.save_success(url, html, content_type, status)
        for linked_url in _diversify_urls(_extract_links(url, html)):
            if linked_url not in state.seen_urls:
                queue.append(linked_url)
        state.save_frontier(queue)
        if len(state.manifest["saved"]) % 10 == 0:
            state.print_summary(len(queue))
    state.save_frontier(queue)
    if max_pages is not None and len(state.manifest["saved"]) >= max_pages:
        print(f"[done] reached max_pages={max_pages}", flush=True)
    elif not queue:
        print("[done] queue exhausted", flush=True)
    else:
        print("[done] crawl stopped with pending frontier", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--sitemap-index-url", action="append", default=[])
    parser.add_argument("--crawl-delay-sec", type=float, default=DEFAULT_CRAWL_DELAY_SEC)
    return parser.parse_args()


def _resolve_sitemap_index_urls(start_urls: list[str], explicit_sitemap_index_urls: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    candidates = list(explicit_sitemap_index_urls)
    if not candidates:
        for start_url in start_urls:
            host = urllib.parse.urlparse(start_url).netloc
            candidates.append(f"https://{host}/sitemap_index.xml")
    for candidate in candidates:
        canonical = canonicalize_url(candidate)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


if __name__ == "__main__":
    args = _parse_args()
    crawl_site(
        start_urls=args.start_url,
        out_dir=args.out,
        max_pages=args.max_pages,
        sitemap_index_urls=args.sitemap_index_url,
        crawl_delay_sec=args.crawl_delay_sec,
    )
