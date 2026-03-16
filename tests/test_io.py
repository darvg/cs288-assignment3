from __future__ import annotations

import tempfile
import unittest

from src.io_utils import read_questions_txt, write_answers_txt


class IOTests(unittest.TestCase):
    def test_read_and_write_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/answers.txt"
            write_answers_txt(path, ["first", "second line"])
            self.assertEqual(read_questions_txt(path), ["first", "second line"])


if __name__ == "__main__":
    unittest.main()
