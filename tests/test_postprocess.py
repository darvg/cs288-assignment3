from __future__ import annotations

import unittest

from src.answer_postprocess import ensure_single_line, fallback_if_empty


class PostprocessTests(unittest.TestCase):
    def test_single_line(self) -> None:
        self.assertEqual(ensure_single_line("hello\nworld"), "hello world")

    def test_fallback(self) -> None:
        self.assertEqual(fallback_if_empty("   "), "unknown")


if __name__ == "__main__":
    unittest.main()
