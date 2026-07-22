from __future__ import annotations

import argparse
import asyncio
import threading
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .image_pipeline import extract_printed_question, process_image
from .pdf_export import export_pdf
from .recognition import RecognitionService
from .storage import Storage

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MAX_FILE_SIZE = 30 * 1024 * 1024
MAX_FILES = 50


class ReviewRequest(BaseModel):
    action: Literal[
        "accept_cleaned",
        "accept_normalized",
        "exclude",
        "set_category",
    ]
    category: str | None = Field(default=None, max_length=24)
    ocr_text: str | None = Field(default=None, max_length=20000)


class ExportRequest(BaseModel):
    allow_partial: bool = False


class Processor:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.recognition = RecognitionService(settings)

    def process_batch(self, batch_id: str) -> None:
        for problem in self.storage.get_problems(batch_id):
            self.process_problem(problem["id"])
        self.storage.finish_batch(batch_id)

    def process_problem(self, problem_id: str) -> None:
        problem = self.storage.get_problem(problem_id)
        if problem is None:
            return
        self.storage.update_problem(problem_id, status="processing", error=None)
        artifact_dir = (
            self.settings.data_dir / "files" / problem["batch_id"] / problem_id
        )
        try:
            source_path = Path(problem["source_path"])
            rotation_hint = self.recognition.rotation_hint(source_path)
            pipeline = process_image(source_path, artifact_dir, rotation_hint)
            recognition = self.recognition.recognize(
                pipeline.normalized_path,
                self.storage.category_names(),
            )
            extraction_reasons: list[str] = []
            if recognition.lines:
                selected, extraction_metrics, extraction_reasons = extract_printed_question(
                    pipeline.cleaned_path,
                    recognition.lines,
                    artifact_dir,
                )
                pipeline.selected_path = selected
                pipeline.metrics.update(extraction_metrics)
            reasons = [
                *pipeline.review_reasons,
                *recognition.review_reasons,
                *extraction_reasons,
            ]
            category = recognition.category
            if category:
                self.storage.ensure_category(category, recognition.summary)
            metrics = {
                **pipeline.metrics,
                "recognition_provider": recognition.provider,
                "recognition_summary": recognition.summary,
                "review_reasons": reasons,
            }
            status = "needs_review" if reasons else "ready"
            review_status = "pending" if reasons else "not_required"
            self.storage.update_problem(
                problem_id,
                status=status,
                review_status=review_status,
                selected_artifact=str(pipeline.selected_path),
                category=category,
                ocr_text=recognition.text,
                confidence=recognition.confidence,
                metrics_json=metrics,
            )
        except Exception as error:  # 单题失败不能中断批次
            self.storage.update_problem(
                problem_id,
                status="failed",
                review_status="pending",
                error=f"{type(error).__name__}: {error}",
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    storage = Storage(settings.data_dir)
    processor = Processor(settings, storage)
    app = FastAPI(title="小奥错题集", version="0.1.0")
    app.state.settings = settings
    app.state.storage = storage
    app.state.processor = processor

    @app.middleware("http")
    async def local_security(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = request.headers.get("host", "").split(":")[0].strip("[]")
        allowed_hosts = {"127.0.0.1", "localhost", "::1", "testserver"}
        if host not in allowed_hosts:
            return JSONResponse({"detail": "默认只允许本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and not any(
            origin.startswith(prefix)
            for prefix in ("http://127.0.0.1", "http://localhost", "http://[::1]")
        ):
            return JSONResponse({"detail": "拒绝跨站请求"}, status_code=403)
        if request.url.path.startswith("/api/"):
            token = request.headers.get("x-session-token") or request.query_params.get("token")
            if token != settings.session_token:
                return JSONResponse({"detail": "会话令牌无效"}, status_code=403)
        return await call_next(request)

    def public_problem(problem: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value
            for key, value in problem.items()
            if key not in {"source_path", "selected_artifact"}
        }
        result["images"] = {
            "source": f"/api/problems/{problem['id']}/image/source?token={settings.session_token}",
            "normalized": f"/api/problems/{problem['id']}/image/normalized?token={settings.session_token}",
            "cleaned": f"/api/problems/{problem['id']}/image/question?token={settings.session_token}",
        }
        return result

    def public_batch(batch: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in batch.items() if key != "pdf_path"}
        result["problems"] = [public_problem(item) for item in batch["problems"]]
        result["download_url"] = (
            f"/api/batches/{batch['id']}/pdf?token={settings.session_token}"
            if batch.get("pdf_path")
            else None
        )
        return result

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (settings.static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__SESSION_TOKEN__", settings.session_token))

    @app.post("/api/batches")
    async def create_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not files or len(files) > MAX_FILES:
            raise HTTPException(400, f"每批必须包含 1 至 {MAX_FILES} 张图片")
        batch_id = storage.create_batch()
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(400, f"不支持的图片格式：{suffix or '未知'}")
            content = await upload.read(MAX_FILE_SIZE + 1)
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"{upload.filename} 超过 30 MiB")
            storage.add_uploaded_problem(batch_id, upload.filename or "image", content)
        threading.Thread(
            target=processor.process_batch,
            args=(batch_id,),
            daemon=True,
            name=f"batch-{batch_id[:8]}",
        ).start()
        batch = storage.get_batch(batch_id)
        assert batch is not None
        return public_batch(batch)

    @app.post("/api/batches/import-sample")
    async def import_sample() -> dict[str, Any]:
        sample_dir = settings.root_dir / "Sample"
        images = sorted(
            path
            for path in sample_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
        )
        if not images:
            raise HTTPException(404, "Sample 目录中没有支持的图片")
        batch_id = storage.create_batch()
        for image in images[:MAX_FILES]:
            storage.add_problem(batch_id, image.name, image)
        threading.Thread(
            target=processor.process_batch,
            args=(batch_id,),
            daemon=True,
            name=f"sample-{batch_id[:8]}",
        ).start()
        batch = storage.get_batch(batch_id)
        assert batch is not None
        return public_batch(batch)

    @app.get("/api/batches/{batch_id}")
    async def get_batch(batch_id: str) -> dict[str, Any]:
        batch = storage.get_batch(batch_id)
        if batch is None:
            raise HTTPException(404, "批次不存在")
        return public_batch(batch)

    @app.get("/api/problems/{problem_id}/image/{kind}")
    async def problem_image(problem_id: str, kind: str) -> FileResponse:
        problem = storage.get_problem(problem_id)
        if problem is None:
            raise HTTPException(404, "题目不存在")
        if kind == "source":
            path = Path(problem["source_path"])
        elif kind in {"normalized", "cleaned", "question"}:
            candidate = (
                settings.data_dir
                / "files"
                / problem["batch_id"]
                / problem_id
                / f"{kind}.png"
            )
            path = candidate
            if kind == "question" and not path.exists():
                path = candidate.with_name("cleaned.png")
        else:
            raise HTTPException(404, "图片类型不存在")
        if not path.exists():
            raise HTTPException(404, "图片尚未生成")
        return FileResponse(path)

    @app.post("/api/problems/{problem_id}/review")
    async def review_problem(problem_id: str, payload: ReviewRequest) -> dict[str, Any]:
        problem = storage.get_problem(problem_id)
        if problem is None:
            raise HTTPException(404, "题目不存在")
        artifact_dir = (
            settings.data_dir / "files" / problem["batch_id"] / problem_id
        )
        values: dict[str, Any]
        if payload.action == "exclude":
            values = {"review_status": "excluded", "status": "excluded"}
        elif payload.action in {"accept_cleaned", "accept_normalized"}:
            kind = payload.action.removeprefix("accept_")
            target = artifact_dir / f"{kind}.png"
            if kind == "cleaned" and (artifact_dir / "question.png").exists():
                target = artifact_dir / "question.png"
            if not target.exists():
                raise HTTPException(409, "指定产物尚未生成")
            values = {
                "selected_artifact": str(target),
                "review_status": "accepted",
                "status": "ready",
            }
        else:
            category = (payload.category or "").strip() or "未分类"
            storage.ensure_category(category)
            values = {
                "category": category,
                "ocr_text": payload.ocr_text or problem.get("ocr_text") or "人工确认题目",
                "review_status": "accepted",
                "status": "ready",
            }
        storage.update_problem(problem_id, **values)
        storage.finish_batch(problem["batch_id"])
        updated = storage.get_problem(problem_id)
        assert updated is not None
        return public_problem(updated)

    @app.post("/api/batches/{batch_id}/export")
    async def create_pdf(batch_id: str, payload: ExportRequest) -> dict[str, str]:
        batch = storage.get_batch(batch_id)
        if batch is None:
            raise HTTPException(404, "批次不存在")
        try:
            path = await asyncio.to_thread(
                export_pdf,
                batch_id,
                batch["problems"],
                settings.data_dir / "exports",
                allow_partial=payload.allow_partial,
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        storage.set_pdf(batch_id, path)
        return {
            "download_url": f"/api/batches/{batch_id}/pdf?token={settings.session_token}"
        }

    @app.get("/api/batches/{batch_id}/pdf")
    async def download_pdf(batch_id: str) -> FileResponse:
        batch = storage.get_batch(batch_id)
        if batch is None or not batch.get("pdf_path"):
            raise HTTPException(404, "PDF 尚未生成")
        path = Path(batch["pdf_path"])
        if not path.exists():
            raise HTTPException(404, "PDF 文件不存在")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地小奥错题集工具")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("第一版仅允许监听本机地址")
    settings = Settings.load(args.root)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
