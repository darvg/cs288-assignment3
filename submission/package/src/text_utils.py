from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def simple_tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def tokenize_with_counts(text: str) -> Counter[str]:
    return Counter(simple_tokenize(text))


def split_words(text: str) -> list[str]:
    return normalize_whitespace(text).split()
