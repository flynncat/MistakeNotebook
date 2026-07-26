from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Install packages without downloading the default OCR models.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    venv = root / ".models" / "paddleocr-venv"
    python = _venv_python(venv)
    if sys.platform == "darwin" and platform.machine() != "arm64":
        raise RuntimeError(
            "PaddlePaddle 3.3 no longer provides Intel macOS wheels. "
            "Use MISTAKE_BOOK_LOCAL_OCR=vision on Intel Mac."
        )
    if not python.exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "venv", str(venv)])
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    paddle_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "paddlepaddle==3.3.0",
    ]
    if sys.platform == "darwin":
        paddle_command.extend(
            ["-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"]
        )
    _run(paddle_command)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "paddleocr>=3.3,<4",
            "pillow>=10.4,<12",
        ]
    )
    if not args.skip_warmup:
        _run(
            [
                str(python),
                str(root / "scripts" / "paddle_ocr_worker.py"),
                "--warmup",
            ]
        )
    print(f"PaddleOCR runtime is ready: {venv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
