from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


UNIMERNET_REVISION = "3f09ac4b1cd583be47ea20a7d7daef839473028a"
MFD_REVISION = "f470a885e0fca1d3d2bfa2a54991db7ae01f1861"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _ensure_venv(path: Path, packages: list[str]) -> None:
    python = (
        path / "Scripts" / "python.exe"
        if os.name == "nt"
        else path / "bin" / "python"
    )
    if not python.exists():
        _run([sys.executable, "-m", "venv", str(path)])
    _run([str(python), "-m", "pip", "install", *packages])


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                with temporary.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    temporary.replace(target)


def _manifest_models(root: Path) -> dict[str, dict[str, object]]:
    payload = json.loads((root / "models" / "v2_manifest.json").read_text("utf-8"))
    return {str(item["id"]): item for item in payload["models"]}


def _verify(path: Path, metadata: dict[str, object]) -> None:
    expected_size = int(metadata["weight_bytes"])
    expected_hash = str(metadata["weight_sha256"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"{path.name} has {path.stat().st_size} bytes, expected {expected_size}"
        )
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{path.name} SHA-256 is {actual_hash}, expected {expected_hash}"
        )


def _matches(path: Path, metadata: dict[str, object]) -> bool:
    try:
        _verify(path, metadata)
    except (FileNotFoundError, RuntimeError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    models_root = root / ".models"
    manifest = _manifest_models(root)
    if not args.skip_runtime:
        _ensure_venv(
            models_root / "unimernet-venv",
            ["unimernet==0.2.3", "huggingface-hub"],
        )
        _ensure_venv(
            models_root / "pix2text-venv",
            ["pix2text==1.1.6"],
        )

    tiny_dir = models_root / "weights" / "unimernet_tiny"
    tiny_files = (
        "README.md",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "unimernet_tiny.pth",
    )
    for filename in tiny_files:
        target = tiny_dir / filename
        if (
            filename == "unimernet_tiny.pth"
            and target.exists()
            and not _matches(target, manifest["unimernet-tiny"])
        ):
            target.unlink()
        if target.exists():
            continue
        _download(
            "https://huggingface.co/wanderkid/unimernet_tiny/resolve/"
            f"{UNIMERNET_REVISION}/{filename}?download=true",
            target,
        )

    mfd_path = (
        models_root
        / "weights"
        / "pix2text-mfd-1.5"
        / "pix2text-mfd-1.5.onnx"
    )
    if mfd_path.exists() and not _matches(
        mfd_path,
        manifest["pix2text-mfd-1.5"],
    ):
        mfd_path.unlink()
    if not mfd_path.exists():
        _download(
            "https://huggingface.co/breezedeus/pix2text-mfd-1.5/resolve/"
            f"{MFD_REVISION}/pix2text-mfd-1.5.onnx?download=true",
            mfd_path,
        )

    _verify(tiny_dir / "unimernet_tiny.pth", manifest["unimernet-tiny"])
    _verify(mfd_path, manifest["pix2text-mfd-1.5"])
    print("Formula models and isolated runtimes are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
