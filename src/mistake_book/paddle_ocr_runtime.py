from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

from .runtime_paths import venv_python


class PaddleOCRRuntimeError(RuntimeError):
    pass


class PaddleOCRRuntime:
    def __init__(self, project_root: Path, data_dir: Path) -> None:
        self.project_root = project_root.resolve()
        self.python = venv_python(
            self.project_root / ".models" / "paddleocr-venv"
        )
        self.worker_script = (
            self.project_root / "scripts" / "paddle_ocr_worker.py"
        )
        self.log_path = data_dir / "paddle-ocr.log"
        self._process: subprocess.Popen[str] | None = None
        self._log = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self._closed = False

    @property
    def available(self) -> bool:
        return self.python.is_file() and self.worker_script.is_file()

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        lines: queue.Queue[str | None],
    ) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    def _readline(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as error:
            raise PaddleOCRRuntimeError("PaddleOCR worker response timed out") from error
        if line is None:
            code = self._process.poll() if self._process is not None else None
            raise PaddleOCRRuntimeError(
                f"PaddleOCR worker exited with code {code}"
            )
        return line

    def _start(self) -> None:
        if self._closed:
            raise PaddleOCRRuntimeError("PaddleOCR runtime is closed")
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        if not self.available:
            raise PaddleOCRRuntimeError(
                "PaddleOCR is not installed; run scripts/setup_paddle_ocr.py"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        self._lines = queue.Queue()
        try:
            self._process = subprocess.Popen(
                [str(self.python), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                text=True,
                bufsize=1,
            )
            self._reader = threading.Thread(
                target=self._read_stdout,
                args=(self._process, self._lines),
                daemon=True,
            )
            self._reader.start()
            ready = json.loads(self._readline(300))
        except Exception as error:
            self._stop()
            raise PaddleOCRRuntimeError(
                f"PaddleOCR worker failed to start: {error}"
            ) from error
        if ready.get("ready") is not True:
            self._stop()
            raise PaddleOCRRuntimeError("PaddleOCR worker did not become ready")

    def recognize(self, image_path: Path) -> list[dict[str, Any]]:
        with self._lock:
            self._start()
            assert self._process is not None
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(
                    json.dumps(
                        {"image_path": str(image_path.resolve())},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._process.stdin.flush()
                response = json.loads(self._readline(180))
            except Exception as error:
                self._stop()
                if isinstance(error, PaddleOCRRuntimeError):
                    raise
                raise PaddleOCRRuntimeError(
                    f"PaddleOCR request failed: {error}"
                ) from error
            if response.get("ok") is not True:
                raise PaddleOCRRuntimeError(
                    str(response.get("error") or "PaddleOCR request failed")
                )
            return [
                dict(line)
                for line in response.get("lines") or []
                if isinstance(line, dict)
            ]

    def _stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._log is not None:
            self._log.close()
            self._log = None
        self._reader = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop()
