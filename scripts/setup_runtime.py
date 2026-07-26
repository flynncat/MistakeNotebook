from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _run(script: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=script.parent.parent,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install document, formula, and cross-platform OCR runtimes."
    )
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-formulas", action="store_true")
    parser.add_argument("--skip-paddle", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if not args.skip_v2:
        _run(scripts / "setup_v2_models.py")
    if not args.skip_formulas:
        _run(scripts / "setup_formula_models.py")
    if not args.skip_paddle:
        paddle_arguments = ("--skip-warmup",) if args.skip_warmup else ()
        _run(scripts / "setup_paddle_ocr.py", *paddle_arguments)
    print("Mistake Notebook runtimes are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
