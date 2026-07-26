from __future__ import annotations

import os
from pathlib import Path


def venv_python(venv: Path, os_name: str | None = None) -> Path:
    if (os_name or os.name) == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"
