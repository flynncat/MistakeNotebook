from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

from mistake_book.config import Settings
from mistake_book.recognition import PaddleOCRLocal, RecognitionService
from mistake_book.runtime_paths import venv_python


def _paddle_worker_class():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paddle_ocr_worker.py"
    specification = importlib.util.spec_from_file_location(
        "paddle_ocr_worker_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.PaddleOCRWorker


def test_venv_python_uses_platform_specific_layout(tmp_path: Path) -> None:
    venv = tmp_path / "runtime"

    assert venv_python(venv, "posix") == venv / "bin" / "python"
    assert venv_python(venv, "nt") == venv / "Scripts" / "python.exe"


def test_non_macos_auto_backend_selects_paddle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTAKE_BOOK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("mistake_book.recognition.sys.platform", "linux")
    service = RecognitionService(Settings.load(tmp_path))
    try:
        assert isinstance(service.local_ocr, PaddleOCRLocal)
    finally:
        service.close()


def test_paddle_worker_converts_pixels_to_vision_coordinates(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "question.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class Result:
        res = {
            "rec_texts": ["\u3010\u7ec3\u4e601\u3011"],
            "rec_scores": [0.92],
            "rec_polys": [
                [[20, 10], [120, 10], [120, 30], [20, 30]],
            ],
        }

    class Engine:
        @staticmethod
        def predict(**_kwargs):
            return [Result()]

    worker_class = _paddle_worker_class()
    worker = worker_class.__new__(worker_class)
    worker.engine = Engine()

    payload = worker.recognize(image_path)

    assert payload["lines"] == [
        {
            "text": "\u3010\u7ec3\u4e601\u3011",
            "confidence": 0.92,
            "box": [0.1, 0.7, 0.5, 0.2],
        }
    ]
