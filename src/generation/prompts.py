from __future__ import annotations


def select_prompt_contexts(question: str, retrieved: list, max_contexts: int = 4) -> list[dict]:
    ranked = sorted(retrieved, key=lambda chunk: _context_priority(question, chunk), reverse=True)
    contexts: list[dict] = []
    for chunk in ranked[:max_contexts]:
        contexts.append(
            {
                "title": chunk.title,
                "url": chunk.url,
                "heading_path": chunk.heading_path,
                "text": chunk.text,
            }
        )
    return contexts


def _context_priority(question: str, chunk) -> tuple[float, float]:
    q = question.lower()
    title = chunk.title.lower()
    url = chunk.url.lower()
    score = float(chunk.score)
    penalty = 0.0
    if "archives - page" in title or "/page/" in url or "/category/" in url or "/tag/" in url:
        penalty -= 2.0
    if q.startswith("who "):
        for token in q.replace("?", "").split()[2:]:
            if token and token in title:
                score += 0.5
            if token and token in chunk.text.lower():
                score += 0.1
    if q.startswith("which eecs ") and " page covers " in q:
        target = q.split(" page covers ", 1)[1].replace("?", "")
        for token in target.split():
            if token and token in title:
                score += 0.75
            if token and token in url:
                score += 0.5
    return (score + penalty, chunk.score)


def build_qa_prompt(question: str, contexts: list[dict]) -> str:
    sections = [
        "Answer the question using only the provided contexts.",
        "Return only the short answer. If unsupported, return unknown.",
        f"Question: {question}",
        "Contexts:",
    ]
    for index, context in enumerate(contexts, start=1):
        sections.append(
            f"[{index}] Title: {context['title']} | URL: {context['url']} | Heading: {context['heading_path']}\n"
            f"{context['text']}"
        )
    return "\n\n".join(sections)
