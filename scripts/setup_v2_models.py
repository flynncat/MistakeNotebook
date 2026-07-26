from __future__ import annotations

import subprocess
from pathlib import Path

from mistake_book.v2_models import load_manifest, sha256


REPOSITORIES = {
    "uvdoc": (
        "https://github.com/tanguymagne/uvdoc.git",
        "4c9b82b537057aff2526e6dd118a847cdd072e82",
    ),
}


def _run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def _checkout(root: Path, name: str, url: str, commit: str) -> Path:
    target = root / ".models" / "sources" / name
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", url, str(target))
    _run("git", "fetch", "--depth", "1", "origin", commit, cwd=target)
    _run("git", "checkout", "--detach", commit, cwd=target)
    return target


def main() -> None:
    root = Path.cwd().resolve()
    manifest = load_manifest(root)
    by_id = {item["id"]: item for item in manifest["models"]}
    for name, (url, commit) in REPOSITORIES.items():
        path = _checkout(root, name, url, commit)
        print(f"{name}: {path}")

    checks = {
        root / ".models" / "sources" / "uvdoc" / "model" / "best_model.pkl": by_id[
            "uvdoc"
        ]["weight_sha256"],
    }
    for path, expected in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{path} 哈希不匹配：{actual}")
        print(f"verified: {path}")


if __name__ == "__main__":
    main()
