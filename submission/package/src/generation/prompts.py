from __future__ import annotations


def select_prompt_contexts(retrieved: list, max_contexts: int = 4) -> list[dict]:
    contexts: list[dict] = []
    for chunk in retrieved[:max_contexts]:
        contexts.append(
            {
                "title": chunk.title,
                "url": chunk.url,
                "heading_path": chunk.heading_path,
                "text": chunk.text,
            }
        )
    return contexts


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
