from __future__ import annotations

import unittest

from src.eval.metrics import exact_match, token_f1


class MetricsTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(exact_match("773 Soda Hall", ["773 soda hall"]), 1.0)

    def test_token_f1(self) -> None:
        self.assertGreater(token_f1("Berkeley EECS", ["EECS Berkeley"]), 0.9)


if __name__ == "__main__":
    unittest.main()
