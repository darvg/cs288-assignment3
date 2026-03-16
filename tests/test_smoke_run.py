from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SmokeRunTests(unittest.TestCase):
    def test_run_sh_writes_matching_lines(self) -> None:
        output_path = Path(tempfile.gettempdir()) / "cs288_smoke_predictions.txt"
        if output_path.exists():
            output_path.unlink()
        subprocess.run(
            ["bash", "run.sh", "data/sample/questions.txt", str(output_path)],
            cwd=ROOT,
            check=True,
        )
        questions = (ROOT / "data/sample/questions.txt").read_text(encoding="utf-8").splitlines()
        predictions = output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(questions), len(predictions))
        self.assertTrue(all("\n" not in line for line in predictions))


if __name__ == "__main__":
    unittest.main()
