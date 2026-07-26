from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import Settings
from .content_blocks import build_content_blocks, source_sha256
from .docx_export import export_docx
from .formula_math import FormulaValidationError, convert_latex
from .image_pipeline import extract_printed_question, process_image
from .markdown_export import (
    export_markdown,
    markdown_exportable,
)
from .page_segmentation import PageSegmenter
from .pdf_export import export_pdf
from .recognition import RecognitionService
from .reconstruction import (
    build_structured_problem,
    content_blocks_to_text,
    question_content_blocks,
    render_problem,
    render_problem_content_blocks,
)
from .review_diagnostics import build_review_diagnostics
from .storage import Storage
from .taxonomy import TaxonomyConflict, TaxonomyError, TaxonomyService
from .v2_pipeline import V2Processor

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MAX_FILE_SIZE = 30 * 1024 * 1024
MAX_FILES = 200
MAX_TAXONOMY_BODY = 256 * 1024


class ReviewRequest(BaseModel):
    action: Literal[
        "accept_cleaned",
        "accept_normalized",
        "exclude",
        "set_category",
    ]
    category_group: str | None = Field(default=None, max_length=12)
    category: str | None = Field(default=None, max_length=24)
    ocr_text: str | None = Field(default=None, max_length=20000)


class ExportRequest(BaseModel):
    allow_partial: bool = False


class FormulaPreviewRequest(BaseModel):
    latex: str = Field(min_length=1, max_length=8192)


class FormulaEditItem(BaseModel):
    formula_id: str = Field(min_length=1, max_length=100)
    mode: Literal["latex", "image"] = "latex"
    latex: str | None = Field(default=None, max_length=8192)


class FormulaEditRequest(BaseModel):
    updated_at: str = Field(min_length=1, max_length=64)
    formulas: list[FormulaEditItem] = Field(min_length=1, max_length=64)


class Processor:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        taxonomy: TaxonomyService | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.taxonomy = taxonomy or TaxonomyService(
            settings.data_dir,
            storage.category_pairs(),
        )
        self.recognition = RecognitionService(settings, self.taxonomy.active_builtin)
        self.v2 = V2Processor(settings, self.recognition)
        self.page_segmenter = PageSegmenter(
            self.recognition.local_ocr,
            self.v2.formulas,
        )

    def process_batch(self, batch_id: str) -> None:
        for problem in self.storage.get_problems(batch_id):
            child_ids = self._split_problem(problem)
            if child_ids:
                for child_id in child_ids:
                    self.process_problem(child_id)
            else:
                self.process_problem(problem["id"])
        self.storage.finish_batch(batch_id)

    def _split_problem(self, problem: dict[str, Any]) -> list[str]:
        artifact_dir = (
            self.settings.data_dir
            / "files"
            / problem["batch_id"]
            / problem["id"]
            / "page-split"
        )
        try:
            split = self.page_segmenter.split(
                Path(problem["source_path"]),
                artifact_dir,
                self.recognition.rotation_hint(Path(problem["source_path"])),
            )
        except Exception as error:
            metrics = dict(problem.get("metrics") or {})
            metrics["page_split"] = {
                "attempted": True,
                "error": f"{type(error).__name__}: {error}",
            }
            self.storage.update_problem(problem["id"], metrics_json=metrics)
            return []
        if split is None:
            return []

        total = len(split.regions)
        stem = Path(problem["filename"]).stem
        relative_path = problem.get("source_relative_path") or problem["filename"]
        child_ids: list[str] = []
        for index, region in enumerate(split.regions, start=1):
            child_id = self.storage.add_problem(
                problem["batch_id"],
                f"{stem}-{index:02d}.png",
                region.source_path,
                source_relative_path=f"{relative_path}#{region.label}",
                parent_source_id=problem["id"],
                page_index=index,
                page_total=total,
                split_label=region.label,
                split_ocr_text=region.ocr_text,
            )
            child_metrics = {
                "page_split": {
                    **split.metrics,
                    "region": region.metrics(),
                    "page_index": index,
                    "page_total": total,
                }
            }
            self.storage.update_problem(child_id, metrics_json=child_metrics)
            child_ids.append(child_id)
        self.storage.delete_problem(problem["id"])
        return child_ids

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
            if self.settings.pipeline_version == "v2":
                result = self.v2.process(
                    source_path,
                    artifact_dir,
                    self.taxonomy.active_category_names(),
                    rotation_hint,
                    title_hint=problem.get("split_label") or None,
                    primary_text_hint=problem.get("split_ocr_text") or None,
                )
                recognition = result.recognition
                reasons = list(result.review_reasons)
                page_split = (problem.get("metrics") or {}).get("page_split", {})
                region = page_split.get("region", {})
                if region and float(region.get("blur_score", 0)) < 80:
                    reasons.append("该题局部清晰度不足，数字和运算符需要人工核对")
                if page_split.get("error"):
                    reasons.append("检测到疑似多题页面，但自动切分未通过完整性校验")
                with self.taxonomy.mutation_guard():
                    category_group = recognition.category_group
                    category = recognition.category
                    if not self.taxonomy.is_active_pair(category_group, category):
                        category_group = "未分类"
                        category = "未分类"
                        reasons.append("自动识别分类已停用，请手动选择分类")
                    if category:
                        self.storage.ensure_category(category, recognition.summary)
                    generated_blocks = result.metrics.get("content_blocks")
                    previous_blocks = problem.get("content_blocks")
                    if (
                        isinstance(generated_blocks, dict)
                        and generated_blocks.get("version") == 2
                        and isinstance(previous_blocks, dict)
                        and previous_blocks.get("version") == 2
                    ):
                        previous_formulas = {
                            str(block.get("formula_id")): block
                            for block in previous_blocks.get("blocks") or []
                            if block.get("type") == "latex"
                            and block.get("formula_id")
                        }
                        archived = [
                            dict(block)
                            for block in previous_formulas.values()
                            if block.get("recognition_state")
                            in {"human_verified", "human_verified_image"}
                        ]
                        for block in generated_blocks.get("blocks") or []:
                            previous = previous_formulas.get(
                                str(block.get("formula_id") or "")
                            )
                            if (
                                previous
                                and previous.get("recognition_state")
                                in {"human_verified", "human_verified_image"}
                            ):
                                block["latex"] = previous.get("latex")
                                block["recognition_state"] = previous.get(
                                    "recognition_state"
                                )
                                block["edited_at"] = previous.get("edited_at")
                        if archived:
                            result.metrics["previous_human_formula_blocks"] = archived
                    metrics = {
                        **(problem.get("metrics") or {}),
                        **result.metrics,
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
                        selected_artifact=str(result.selected_path),
                        category_group=category_group,
                        category=category,
                        category_key=category,
                        summary=recognition.summary,
                        category_confidence=recognition.category_confidence,
                        category_source=recognition.category_source,
                        ocr_text=result.structured.primary_text,
                        confidence=recognition.confidence,
                        metrics_json=metrics,
                        content_blocks_version=int(
                            (result.metrics.get("content_blocks") or {}).get(
                                "version",
                                1,
                            )
                        ),
                        content_blocks_json=result.metrics.get("content_blocks"),
                        content_source_sha256=result.metrics.get(
                            "content_source_sha256"
                        ),
                    )
                return
            pipeline = process_image(source_path, artifact_dir, rotation_hint)
            recognition = self.recognition.recognize(
                pipeline.normalized_path,
                self.taxonomy.active_category_names(),
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
            with self.taxonomy.mutation_guard():
                category_group = recognition.category_group
                category = recognition.category
                if not self.taxonomy.is_active_pair(category_group, category):
                    category_group = "未分类"
                    category = "未分类"
                    reasons.append("自动识别分类已停用，请手动选择分类")
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
                    category_group=category_group,
                    category=category,
                    category_key=category,
                    summary=recognition.summary,
                    category_confidence=recognition.category_confidence,
                    category_source=recognition.category_source,
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
    taxonomy = TaxonomyService(settings.data_dir, storage.category_pairs())
    processor = Processor(settings, storage, taxonomy)
    app = FastAPI(title="小奥错题集", version="0.1.0")
    app.state.settings = settings
    app.state.storage = storage
    app.state.taxonomy = taxonomy
    app.state.processor = processor
    app.router.add_event_handler("shutdown", processor.v2.formulas.close)
    app.router.add_event_handler("shutdown", processor.recognition.close)

    def resolve_review_metrics(problem: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(problem.get("metrics") or {})
        reasons = metrics.get("review_reasons")
        if isinstance(reasons, list) and reasons:
            previous = metrics.get("resolved_review_reasons")
            resolved = list(previous) if isinstance(previous, list) else []
            metrics["resolved_review_reasons"] = list(
                dict.fromkeys([*resolved, *reasons])
            )
        metrics["review_reasons"] = []
        metrics["human_verified"] = True
        structured = metrics.get("structured_problem")
        if isinstance(structured, dict):
            structured = dict(structured)
            structured["review_reasons"] = []
            metrics["structured_problem"] = structured
        return metrics

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
        version = str(problem.get("updated_at") or "").replace("+", "%2B")
        result["images"] = {
            "source": f"/api/problems/{problem['id']}/image/source?token={settings.session_token}",
            "normalized": f"/api/problems/{problem['id']}/image/normalized?token={settings.session_token}&v={version}",
            "cleaned": f"/api/problems/{problem['id']}/image/question?token={settings.session_token}&v={version}",
        }
        result["tags"] = {
            "category_group": problem.get("category_group") or "未分类",
            "category_key": problem.get("category_key")
            or problem.get("category")
            or "未分类",
            "active": taxonomy.is_active_pair(
                str(problem.get("category_group") or ""),
                str(problem.get("category_key") or problem.get("category") or ""),
            ),
        }
        result["markdown_exportable"] = markdown_exportable(problem)
        content_blocks = problem.get("content_blocks")
        result["formulas"] = []
        if isinstance(content_blocks, dict):
            for block in content_blocks.get("blocks") or []:
                if block.get("type") != "latex" or not block.get("formula_id"):
                    continue
                formula = dict(block)
                original = str(block.get("original_crop_asset") or "")
                clean = str(block.get("clean_crop_asset") or "")
                if original:
                    formula["original_url"] = (
                        f"/api/problems/{problem['id']}/artifact/{original}"
                        f"?token={settings.session_token}&v={version}"
                    )
                if clean:
                    formula["clean_url"] = (
                        f"/api/problems/{problem['id']}/artifact/{clean}"
                        f"?token={settings.session_token}&v={version}"
                    )
                result["formulas"].append(formula)
        if problem.get("review_status") in {"accepted", "excluded"}:
            result["review_diagnostics"] = []
        else:
            result["review_diagnostics"] = build_review_diagnostics(
                problem.get("metrics"),
                problem.get("error"),
            )
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
        imported_count = 0
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(400, f"不支持的图片格式：{suffix or '未知'}")
            content = await upload.read(MAX_FILE_SIZE + 1)
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"{upload.filename} 超过 30 MiB")
            relative_path = upload.filename or "image"
            storage.add_uploaded_problem(
                batch_id,
                Path(relative_path).name,
                content,
                source_relative_path=relative_path,
            )
            imported_count += 1
        if imported_count:
            threading.Thread(
                target=processor.process_batch,
                args=(batch_id,),
                daemon=True,
                name=f"batch-{batch_id[:8]}",
            ).start()
        else:
            storage.finish_batch(batch_id)
        batch = storage.get_batch(batch_id)
        assert batch is not None
        result = public_batch(batch)
        result["import_summary"] = {
            "imported_count": imported_count,
        }
        return result

    @app.get("/api/batches/{batch_id}")
    async def get_batch(batch_id: str) -> dict[str, Any]:
        batch = storage.get_batch(batch_id)
        if batch is None:
            raise HTTPException(404, "批次不存在")
        return public_batch(batch)

    @app.get("/api/categories")
    async def get_categories() -> dict[str, Any]:
        return {"groups": taxonomy.active_payload()}

    @app.get("/api/taxonomy")
    async def get_taxonomy() -> dict[str, Any]:
        return taxonomy.payload(storage.category_usage())

    @app.put("/api/taxonomy")
    async def update_taxonomy(payload: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_TAXONOMY_BODY:
            raise HTTPException(413, "分类配置不能超过 256 KiB")
        try:
            expected_revision = int(payload.get("expected_revision"))
            configuration = {
                "groups": payload.get("groups"),
            }
            taxonomy.update(configuration, expected_revision)
        except (TypeError, ValueError, TaxonomyError) as error:
            if isinstance(error, TaxonomyConflict):
                raise HTTPException(409, str(error)) from error
            raise HTTPException(422, str(error)) from error
        return taxonomy.payload(storage.category_usage())

    @app.get("/api/problems")
    async def list_all_problems(
        sort: Literal["newest", "oldest"] = "newest",
        category_group: str | None = Query(default=None, max_length=12),
        category: str | None = Query(default=None, max_length=24),
        q: str | None = Query(default=None, max_length=100),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if category and (
            not category_group
            or not (
                taxonomy.is_known_pair(category_group, category)
                or (category_group, category) in storage.category_usage()
            )
        ):
            raise HTTPException(422, "题型筛选必须是合法的一级、二级组合")
        result = storage.list_problems(
            sort=sort,
            category_group=category_group,
            category=category,
            query=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )
        result["items"] = [public_problem(item) for item in result["items"]]
        return result

    @app.get("/api/assets")
    async def list_assets(
        sort: Literal["newest", "oldest"] = "newest",
        category_group: str | None = Query(default=None, max_length=12),
        category: str | None = Query(default=None, max_length=24),
        q: str | None = Query(default=None, max_length=100),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if category and (
            not category_group
            or not (
                taxonomy.is_known_pair(category_group, category)
                or (category_group, category) in storage.category_usage()
            )
        ):
            raise HTTPException(422, "题型筛选必须是合法的一级、二级组合")
        result = storage.list_assets(
            sort=sort,
            category_group=category_group,
            category=category,
            query=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )
        result["items"] = [public_problem(item) for item in result["items"]]
        return result

    @app.get("/api/problems/{problem_id}/image/{kind}")
    async def problem_image(problem_id: str, kind: str) -> Response:
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
        else:
            raise HTTPException(404, "图片类型不存在")
        if not path.exists():
            raise HTTPException(404, "图片尚未生成")
        if kind == "source":
            return FileResponse(path)
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise HTTPException(404, "图片尚未生成") from error
        return Response(
            content=content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/problems/{problem_id}/artifact/{asset}")
    async def problem_artifact(problem_id: str, asset: str) -> Response:
        problem = storage.get_problem(problem_id)
        if problem is None:
            raise HTTPException(404, "\u9898\u76ee\u4e0d\u5b58\u5728")
        if (
            Path(asset).name != asset
            or not asset.startswith("formula-")
            or Path(asset).suffix.lower() != ".png"
        ):
            raise HTTPException(404, "\u516c\u5f0f\u56fe\u7247\u4e0d\u5b58\u5728")
        artifact_dir = (
            settings.data_dir / "files" / problem["batch_id"] / problem_id
        ).resolve()
        path = artifact_dir / asset
        if path.is_symlink() or path.resolve().parent != artifact_dir:
            raise HTTPException(404, "\u516c\u5f0f\u56fe\u7247\u4e0d\u5b58\u5728")
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise HTTPException(
                404,
                "\u516c\u5f0f\u56fe\u7247\u4e0d\u5b58\u5728",
            ) from error
        return Response(
            content=content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/formulas/preview")
    async def preview_formula(payload: FormulaPreviewRequest) -> dict[str, str]:
        if len(
            json.dumps(payload.model_dump(), ensure_ascii=False).encode("utf-8")
        ) > 16 * 1024:
            raise HTTPException(413, "LaTeX preview request is too large")
        try:
            converted = await asyncio.wait_for(
                asyncio.to_thread(convert_latex, payload.latex),
                timeout=2,
            )
        except TimeoutError as error:
            raise HTTPException(408, "LaTeX preview timed out") from error
        except FormulaValidationError as error:
            raise HTTPException(422, str(error)) from error
        return {"latex": converted.latex, "mathml": converted.mathml}

    @app.put("/api/problems/{problem_id}/formulas")
    async def update_formulas(
        problem_id: str,
        payload: FormulaEditRequest,
    ) -> dict[str, Any]:
        if len(
            json.dumps(payload.model_dump(), ensure_ascii=False).encode("utf-8")
        ) > 512 * 1024:
            raise HTTPException(413, "Formula update request is too large")
        problem = storage.get_problem(problem_id)
        if problem is None:
            raise HTTPException(404, "\u9898\u76ee\u4e0d\u5b58\u5728")
        content_blocks = problem.get("content_blocks")
        if not (
            isinstance(content_blocks, dict)
            and content_blocks.get("version") == 2
            and isinstance(content_blocks.get("blocks"), list)
        ):
            raise HTTPException(409, "Formula blocks are not available")
        edits = {item.formula_id: item for item in payload.formulas}
        if len(edits) != len(payload.formulas):
            raise HTTPException(422, "Formula IDs must be unique")
        existing = {
            str(block.get("formula_id")): block
            for block in content_blocks["blocks"]
            if block.get("type") == "latex" and block.get("formula_id")
        }
        unknown = sorted(set(edits) - set(existing))
        if unknown:
            raise HTTPException(422, f"Unknown formula ID: {unknown[0]}")
        converted_values: dict[str, tuple[str, str | None]] = {}
        try:
            for formula_id, edit in edits.items():
                if edit.mode == "image":
                    if not existing[formula_id].get("clean_crop_asset"):
                        raise FormulaValidationError(
                            "Formula image is not available"
                        )
                    converted_values[formula_id] = ("image", None)
                    continue
                if not edit.latex:
                    raise FormulaValidationError("LaTeX formula is required")
                converted_values[formula_id] = (
                    "latex",
                    convert_latex(edit.latex).latex,
                )
        except FormulaValidationError as error:
            raise HTTPException(422, str(error)) from error
        now = datetime.now(UTC).isoformat()
        updated_blocks = {
            **content_blocks,
            "blocks": [dict(block) for block in content_blocks["blocks"]],
        }
        for block in updated_blocks["blocks"]:
            formula_id = str(block.get("formula_id") or "")
            if formula_id not in converted_values:
                continue
            mode, latex = converted_values[formula_id]
            if mode == "image":
                block["recognition_state"] = "human_verified_image"
            else:
                block["latex"] = latex
                block["recognition_state"] = "human_verified"
            block["edited_at"] = now
        metrics = dict(problem.get("metrics") or {})
        structured_data = metrics.get("structured_problem")
        title_hint = (
            str(structured_data.get("title") or "")
            if isinstance(structured_data, dict)
            else ""
        )
        structured = build_structured_problem(
            str(problem.get("ocr_text") or ""),
            title_hint=title_hint or None,
        )
        if title_hint:
            structured.title = title_hint
        updated_blocks = question_content_blocks(structured, updated_blocks)
        recognition_metrics = dict(metrics.get("formula_recognition") or {})
        states = [
            block.get("recognition_state")
            for block in updated_blocks["blocks"]
            if block.get("type") == "latex"
        ]
        recognition_metrics["formula_states"] = {
            state: states.count(state)
            for state in (
                "auto_verified",
                "needs_review",
                "human_verified",
                "human_verified_image",
                "image_fallback",
            )
        }
        metrics["formula_recognition"] = recognition_metrics
        reasons = metrics.get("review_reasons")
        if isinstance(reasons, list) and not any(
            state in {"needs_review", "image_fallback"} for state in states
        ):
            metrics["review_reasons"] = [
                reason
                for reason in reasons
                if not str(reason).startswith("Formula ")
            ]
        formula_aware_text = content_blocks_to_text(structured, updated_blocks)
        structured.primary_text = formula_aware_text
        structured.body = formula_aware_text.removeprefix(structured.title).strip()
        metrics["structured_problem"] = structured.to_dict()
        artifact_dir = (
            settings.data_dir / "files" / problem["batch_id"] / problem_id
        )
        pending_question = artifact_dir / f"question-{uuid.uuid4().hex}.pending.png"
        try:
            await asyncio.to_thread(
                render_problem_content_blocks,
                structured,
                updated_blocks,
                artifact_dir,
                pending_question,
            )
        except (OSError, ValueError) as error:
            pending_question.unlink(missing_ok=True)
            raise HTTPException(422, f"公式已识别，但白底题目重建失败：{error}") from error
        if not storage.update_formula_blocks(
            problem_id,
            payload.updated_at,
            updated_blocks,
            metrics,
            formula_aware_text,
        ):
            pending_question.unlink(missing_ok=True)
            raise HTTPException(
                409,
                "\u9898\u76ee\u5df2\u88ab\u5176\u4ed6\u64cd\u4f5c\u66f4\u65b0\uff0c\u8bf7\u5237\u65b0\u540e\u91cd\u8bd5",
            )
        pending_question.replace(artifact_dir / "question.png")
        updated = storage.get_problem(problem_id)
        assert updated is not None
        return public_problem(updated)

    @app.post("/api/problems/{problem_id}/review")
    async def review_problem(problem_id: str, payload: ReviewRequest) -> dict[str, Any]:
        problem = storage.get_problem(problem_id)
        if problem is None:
            raise HTTPException(404, "题目不存在")
        artifact_dir = (
            settings.data_dir / "files" / problem["batch_id"] / problem_id
        )
        values: dict[str, Any]
        selected_kind = ""
        if payload.action == "exclude":
            deleted = storage.delete_problem(problem_id)
            if deleted is None:
                raise HTTPException(404, "题目不存在")
            return {
                "id": problem_id,
                "deleted": True,
                "message": "需重拍记录及其文件已删除",
            }
        content_blocks = problem.get("content_blocks")
        unresolved_formulas = []
        if isinstance(content_blocks, dict):
            unresolved_formulas = [
                block.get("formula_id")
                for block in content_blocks.get("blocks") or []
                if block.get("type") == "latex"
                and block.get("recognition_state")
                in {"needs_review", "image_fallback"}
            ]
        if unresolved_formulas:
            raise HTTPException(
                409,
                "\u8bf7\u5148\u6821\u6b63\u5e76\u4fdd\u5b58\u5f85\u786e\u8ba4\u516c\u5f0f",
            )
        if payload.action in {"accept_cleaned", "accept_normalized"}:
            current_group = str(problem.get("category_group") or "")
            current_category = str(
                problem.get("category_key") or problem.get("category") or ""
            )
            if not taxonomy.is_active_pair(current_group, current_category):
                raise HTTPException(
                    422,
                    "当前一级领域和二级题型已停用或无效，请先重新选择分类",
                )
            kind = payload.action.removeprefix("accept_")
            target = artifact_dir / f"{kind}.png"
            if kind == "cleaned" and (artifact_dir / "question.png").exists():
                target = artifact_dir / "question.png"
            if not target.exists():
                raise HTTPException(409, "指定产物尚未生成")
            selected_kind = "reconstructed" if target.name == "question.png" else kind
            values = {
                "selected_artifact": str(target),
                "review_status": "accepted",
                "status": "ready",
                "metrics_json": resolve_review_metrics(problem),
            }
        else:
            corrected_text = payload.ocr_text or problem.get("ocr_text") or "人工确认题目"
            has_formula_blocks = (
                isinstance(content_blocks, dict)
                and content_blocks.get("version") == 2
                and any(
                    block.get("type") == "latex"
                    for block in content_blocks.get("blocks") or []
                )
            )
            if (
                has_formula_blocks
                and corrected_text != str(problem.get("ocr_text") or "")
            ):
                raise HTTPException(
                    409,
                    "\u542b\u516c\u5f0f\u9898\u76ee\u4e0d\u80fd\u5728\u6574\u6bb5\u9898\u5e72\u4e2d\u6539\u52a8\u6587\u5b57\uff0c\u8bf7\u5148\u4fdd\u5b58\u516c\u5f0f\u6216\u91cd\u65b0\u5904\u7406\u8be5\u9898",
                )
            classification = processor.recognition.classify_text(corrected_text)
            category_group = classification.group
            category = classification.category
            category_confidence = classification.confidence
            category_source = classification.source
            summary = classification.summary
            manual_group = (payload.category_group or "").strip()
            manual_category = (payload.category or "").strip()
            if manual_group or manual_category:
                if not (
                    manual_group
                    and manual_category
                    and taxonomy.is_active_pair(manual_group, manual_category)
                ):
                    raise HTTPException(422, "必须同时提交合法的一级领域和二级题型")
                category_group = manual_group
                category = manual_category
                category_confidence = 1.0
                category_source = "manual"
            if not taxonomy.is_active_pair(category_group, category):
                raise HTTPException(
                    422,
                    "未识别到启用中的分类，请手动选择一级领域和二级题型",
                )
            storage.ensure_category(category, summary)
            selected_artifact = problem.get("selected_artifact")
            if settings.pipeline_version == "v2" and payload.ocr_text:
                structured = build_structured_problem(corrected_text, corrected_text)
                figure_path = artifact_dir / "figure-selected.png"
                selected_artifact = str(
                    render_problem(
                        structured,
                        artifact_dir / "question.png",
                        figure_path=figure_path if figure_path.exists() else None,
                    )
                )
                selected_kind = "reconstructed"
            elif selected_artifact:
                selected_kind = "reconstructed"
            values = {
                "category_group": category_group,
                "category": category,
                "category_key": category,
                "summary": summary,
                "category_confidence": category_confidence,
                "category_source": category_source,
                "ocr_text": corrected_text,
                "selected_artifact": selected_artifact,
                "review_status": "accepted",
                "status": "ready",
                "metrics_json": resolve_review_metrics(problem),
            }
            if settings.pipeline_version == "v2" and payload.ocr_text:
                values["metrics_json"]["structured_problem"] = structured.to_dict()
        block_text = str(values.get("ocr_text") or problem.get("ocr_text") or "")
        figure_path = artifact_dir / "figure-selected.png"
        figure_metrics = (problem.get("metrics") or {}).get(
            "figure_preservation", {}
        )
        existing_blocks = problem.get("content_blocks")
        existing_v2 = (
            isinstance(existing_blocks, dict)
            and existing_blocks.get("version") == 2
            and isinstance(existing_blocks.get("blocks"), list)
        )
        if existing_v2 and block_text == str(problem.get("ocr_text") or ""):
            content_blocks = existing_blocks
            content_blocks_version = 2
        else:
            content_blocks = build_content_blocks(
                block_text,
                figure_asset=figure_path.name if figure_path.exists() else None,
                figure_box=(
                    figure_metrics.get("box")
                    if isinstance(figure_metrics, dict)
                    else None
                ),
            )
            content_blocks_version = 1
            if existing_v2:
                preserved_formulas = [
                    dict(block)
                    for block in existing_blocks["blocks"]
                    if block.get("type") == "latex" and block.get("formula_id")
                ]
                images = [
                    block
                    for block in content_blocks["blocks"]
                    if block.get("type") == "image"
                ]
                content_blocks["blocks"] = [
                    block
                    for block in content_blocks["blocks"]
                    if block.get("type") != "image"
                ] + preserved_formulas + images
                content_blocks["version"] = 2
                content_blocks["text_rebuilt_after_review"] = True
                content_blocks_version = 2
        clean_source = artifact_dir / "cleaned.png"
        values.update(
            {
                "content_blocks_version": content_blocks_version,
                "content_blocks_json": content_blocks,
                "content_source_sha256": (
                    source_sha256(clean_source) if clean_source.exists() else None
                ),
            }
        )
        with taxonomy.mutation_guard():
            final_group = str(
                values.get("category_group") or problem.get("category_group") or ""
            )
            final_category = str(
                values.get("category_key")
                or values.get("category")
                or problem.get("category_key")
                or problem.get("category")
                or ""
            )
            if not taxonomy.is_active_pair(final_group, final_category):
                raise HTTPException(
                    422,
                    "当前一级领域和二级题型已停用或无效，请重新选择后再保存",
                )
            if not storage.update_problem_cas(
                problem_id,
                str(problem.get("updated_at") or ""),
                **values,
            ):
                raise HTTPException(
                    409,
                    "\u9898\u76ee\u5df2\u88ab\u5176\u4ed6\u64cd\u4f5c\u66f4\u65b0\uff0c\u8bf7\u5237\u65b0\u540e\u91cd\u8bd5",
                )
            asset_result = storage.publish_asset(
                problem_id,
                selected_kind=selected_kind or "reconstructed",
            )
        storage.finish_batch(problem["batch_id"])
        updated = storage.get_problem(problem_id)
        assert updated is not None
        result = public_problem(updated)
        result["asset_result"] = asset_result
        result["message"] = (
            f"已保存，并替换 {asset_result['replaced_count']} 个旧版本"
            if asset_result["replaced_count"]
            else "已保存到已处理资产"
        )
        return result

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

    @app.post("/api/batches/{batch_id}/export-markdown")
    async def create_markdown(
        batch_id: str,
        payload: ExportRequest,
    ) -> dict[str, str]:
        batch = storage.get_batch(batch_id)
        if batch is None:
            raise HTTPException(404, "批次不存在")
        try:
            result = await asyncio.to_thread(
                export_markdown,
                batch_id,
                batch["problems"],
                settings.data_dir / "exports",
                allow_partial=payload.allow_partial,
            )
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(409, str(error)) from error
        export_id = storage.add_export(
            batch_id,
            "markdown",
            result.path,
            result.filename,
        )
        return {
            "download_url": f"/api/exports/{export_id}?token={settings.session_token}",
            "filename": result.filename,
        }

    @app.post("/api/batches/{batch_id}/export-docx")
    async def create_docx(
        batch_id: str,
        payload: ExportRequest,
    ) -> dict[str, str]:
        batch = storage.get_batch(batch_id)
        if batch is None:
            raise HTTPException(404, "批次不存在")
        try:
            result = await asyncio.to_thread(
                export_docx,
                batch_id,
                batch["problems"],
                settings.data_dir / "exports",
                allow_partial=payload.allow_partial,
            )
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise HTTPException(409, str(error)) from error
        try:
            export_id = storage.add_export(
                batch_id,
                "docx",
                result.path,
                result.filename,
            )
        except Exception:
            result.path.unlink(missing_ok=True)
            raise
        return {
            "download_url": f"/api/exports/{export_id}?token={settings.session_token}",
            "filename": result.filename,
        }

    @app.get("/api/exports/{export_id}")
    async def download_export(export_id: str) -> FileResponse:
        exported = storage.get_export(export_id)
        media_types = {
            "markdown": "application/zip",
            "docx": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
        }
        if exported is None or exported.get("kind") not in media_types:
            raise HTTPException(404, "导出文件不存在")
        kind = str(exported["kind"])
        path = Path(str(exported["file_path"]))
        exports_dir = (settings.data_dir / "exports").resolve()
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise HTTPException(404, "导出文件不存在") from error
        if path.is_symlink() or resolved.parent != exports_dir:
            raise HTTPException(404, "导出文件路径无效")
        return FileResponse(
            resolved,
            media_type=media_types[kind],
            filename=str(exported["filename"]),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地小奥错题集工具")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("默认仅允许监听本机地址")
    settings = Settings.load(args.root)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
