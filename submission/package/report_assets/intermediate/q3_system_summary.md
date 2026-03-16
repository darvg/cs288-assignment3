# System Summary

The runtime uses a hybrid retriever over offline-built sparse and dense artifacts, then selects top contexts for short answer generation. The default configuration disables networked LLM calls and uses deterministic extractive fallback so the sample runtime remains reproducible.
