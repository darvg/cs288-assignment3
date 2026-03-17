from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.upload_cleaned_to_hf import build_metadata, collect_upload_paths


class HuggingFaceExportTests(unittest.TestCase):
    def test_collect_upload_paths_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pages = root / "pages.jsonl"
            chunks = root / "chunks.jsonl"
            benchmark = root / "benchmark.jsonl"
            pages.write_text('{"page_id":"page_0001"}\n{"page_id":"page_0002"}\n', encoding="utf-8")
            chunks.write_text('{"chunk_id":"chunk_0001"}\n', encoding="utf-8")
            benchmark.write_text('{"id":"live_q_0001"}\n{"id":"live_q_0002"}\n', encoding="utf-8")

            uploads = collect_upload_paths(pages, chunks, benchmark)
            metadata = build_metadata(pages, chunks, benchmark)

            self.assertEqual(uploads["data/pages.jsonl"], pages)
            self.assertEqual(uploads["data/chunks.jsonl"], chunks)
            self.assertEqual(uploads["data/qa_live_benchmark.jsonl"], benchmark)
            self.assertEqual(metadata["num_pages"], 2)
            self.assertEqual(metadata["num_chunks"], 1)
            self.assertEqual(metadata["num_benchmark_examples"], 2)


if __name__ == "__main__":
    unittest.main()
