from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable

from .runtime_paths import venv_python


class FormulaRuntimeError(RuntimeError):
    pass


class FormulaRuntimeUnavailable(FormulaRuntimeError):
    pass


class FormulaWorkerResponseError(FormulaRuntimeError):
    pass


def _verify_model(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise FormulaRuntimeUnavailable(f"formula model size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise FormulaRuntimeUnavailable(f"formula model checksum mismatch: {path.name}")


class _PersistentWorker:
    def __init__(
        self,
        command: list[str],
        log_path: Path,
        *,
        startup_timeout: float,
        request_timeout: float,
        preflight: Callable[[], None],
    ) -> None:
        self.command = command
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.preflight = preflight
        self._process: subprocess.Popen[str] | None = None
        self._log = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self._closed = False

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
        process = self._process
        if process is None or process.stdout is None:
            raise FormulaRuntimeError("formula worker is not running")
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("formula worker response timed out") from error
        if line is None:
            raise FormulaRuntimeError(
                f"formula worker exited with code {process.poll()}"
            )
        return line

    def _start(self) -> None:
        if self._closed:
            raise FormulaRuntimeUnavailable("formula worker is closed")
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        try:
            self.preflight()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("a", encoding="utf-8")
            self._lines = queue.Queue()
            self._process = subprocess.Popen(
                self.command,
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
            ready = json.loads(self._readline(self.startup_timeout))
        except FormulaRuntimeError:
            self._stop()
            raise
        except Exception as error:
            self._stop()
            raise FormulaRuntimeUnavailable(
                f"formula worker failed to start: {error}"
            ) from error
        if ready.get("ready") is not True:
            error = ready.get("error", "formula worker failed to initialize")
            self._stop()
            raise FormulaRuntimeUnavailable(str(error))

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise FormulaRuntimeUnavailable("formula worker is closed")
            for attempt in range(2):
                try:
                    self._start()
                    assert self._process is not None
                    assert self._process.stdin is not None
                    self._process.stdin.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )
                    self._process.stdin.flush()
                    try:
                        response = json.loads(self._readline(self.request_timeout))
                    except (json.JSONDecodeError, OSError) as error:
                        raise FormulaRuntimeError(
                            f"invalid formula worker response: {error}"
                        ) from error
                    if response.get("ok") is not True:
                        raise FormulaWorkerResponseError(
                            str(response.get("error", "formula worker failed"))
                        )
                    return response
                except FormulaWorkerResponseError:
                    raise
                except (
                    BrokenPipeError,
                    FormulaRuntimeError,
                    OSError,
                    TimeoutError,
                ):
                    self._stop()
                    if attempt:
                        raise
            raise FormulaRuntimeError("formula worker failed")

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


class FormulaRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._closed = False
        models = self.project_root / ".models"
        worker_script = self.project_root / "scripts" / "formula_worker.py"
        detector_python = venv_python(models / "pix2text-venv")
        detector_model = (
            models
            / "weights"
            / "pix2text-mfd-1.5"
            / "pix2text-mfd-1.5.onnx"
        )
        recognizer_python = venv_python(models / "unimernet-venv")
        recognizer_model = models / "weights" / "unimernet_tiny"
        logs = models / "logs"
        self._detector_paths = (detector_python, detector_model)
        self._recognizer_paths = (
            recognizer_python,
            recognizer_model / "unimernet_tiny.pth",
        )
        self._detector = _PersistentWorker(
            [
                str(detector_python),
                str(worker_script),
                "detect",
                "--model-path",
                str(detector_model),
            ],
            logs / "formula-detector.log",
            startup_timeout=180,
            request_timeout=120,
            preflight=lambda: _verify_model(
                detector_model,
                80_311_115,
                "40d4fc852d99bcbf25a9478897d2f49fbbb8f7fdd6569c088cd1c31386293bd7",
            ),
        )
        self._recognizer = _PersistentWorker(
            [
                str(recognizer_python),
                str(worker_script),
                "recognize",
                "--model-path",
                str(recognizer_model),
            ],
            logs / "formula-recognizer.log",
            startup_timeout=300,
            request_timeout=300,
            preflight=lambda: _verify_model(
                recognizer_model / "unimernet_tiny.pth",
                430_075_701,
                "6f7608624e2d7549c7f0f05fcfbe073ae521328cf70f1d46374d96f9881d7371",
            ),
        )

    @property
    def detector_available(self) -> bool:
        python, model = self._detector_paths
        return (
            python.exists()
            and model.is_file()
            and model.stat().st_size == 80_311_115
        )

    @property
    def recognizer_available(self) -> bool:
        python, model = self._recognizer_paths
        return (
            python.exists()
            and model.is_file()
            and model.stat().st_size == 430_075_701
            and (model.parent / "config.json").is_file()
            and (model.parent / "tokenizer.json").is_file()
        )

    @property
    def available(self) -> bool:
        return self.detector_available and self.recognizer_available

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "detector_available": self.detector_available,
            "recognizer_available": self.recognizer_available,
            "recognizer": "unimernet-tiny",
        }

    def detect(self, image_path: Path) -> dict[str, Any]:
        if self._closed:
            raise FormulaRuntimeUnavailable("formula runtime is closed")
        if not self.detector_available:
            raise FormulaRuntimeUnavailable("formula detector is not installed")
        return self._detector.request(
            {
                "image_path": str(image_path.resolve()),
                "resized_shape": 1280,
                "confidence": 0.55,
            }
        )

    def recognize(self, image_paths: list[Path]) -> dict[str, Any]:
        if self._closed:
            raise FormulaRuntimeUnavailable("formula runtime is closed")
        if not self.recognizer_available:
            raise FormulaRuntimeUnavailable("formula recognizer is not installed")
        started = time.perf_counter()
        response = self._recognizer.request(
            {"image_paths": [str(path.resolve()) for path in image_paths]}
        )
        response["round_trip_ms"] = round(
            (time.perf_counter() - started) * 1000,
            1,
        )
        return response

    def close(self) -> None:
        self._closed = True
        self._detector.close()
        self._recognizer.close()
