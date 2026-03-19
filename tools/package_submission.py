from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "submission" / "package"


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    for name in [
        "run.sh",
        "llm.py",
        "README.md",
        "requirements_runtime.txt",
        "config",
        "src",
        "tools",
        "data/artifacts",
        "data/qa",
        "data/sample",
        "data/interim/chunks.jsonl",
        "report_assets",
    ]:
        source = ROOT / name
        destination = TARGET / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    print(f"Packaged submission at {TARGET}")


if __name__ == "__main__":
    main()
