from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import read_jsonl


def main() -> None:
    rows = read_jsonl("data/qa/qa_validation.jsonl")
    out = Path("report_assets/tables/dataset_preview.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("id,question,answer,source_url\n")
        for row in rows:
            handle.write(f"{row['id']},{row['question']},{row['answers'][0]},{row['source_url']}\n")


if __name__ == "__main__":
    main()
