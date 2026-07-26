from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import unicodedata
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .content_blocks import content_block_rows, content_blocks_for_problem


_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
_PRINT_CSS = """
@media print {
  @page { size: A4; margin: 15mm; }
  body { font-size: 11pt; line-height: 1.5; }
  .mb-answer-space { height: 60mm; }
  .mb-answer-space.mb-long { height: 100mm; }
  .mb-page-break { break-after: page; page-break-after: always; }
  .mb-print-help { display: none !important; }
  img { max-width: 180mm; max-height: 68mm; object-fit: contain; }
}
""".strip() + "\n"


@dataclass(frozen=True)
class MarkdownExportResult:
    path: Path
    filename: str
    note_name: str


def markdown_exportable(problem: dict[str, Any]) -> bool:
    selected = Path(str(problem.get("selected_artifact") or ""))
    return bool(
        problem.get("review_status") == "accepted"
        and problem.get("selected_kind") == "reconstructed"
        and selected.name == "question.png"
        and selected.is_file()
        and problem.get("category_group")
        and problem.get("category_key")
        and problem.get("ocr_text")
        and (problem.get("accepted_at") or problem.get("updated_at"))
    )


def _anchor(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return "mb-" + hashlib.sha256(payload).hexdigest()[:16]


def _wikilink_alias(value: str) -> str:
    return (
        value.replace("|", "\uff5c")
        .replace("]", "\uff3d")
        .replace("#", "\uff03")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _escape_markdown(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "[", "]", "<", ">"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _problem_title(problem: dict[str, Any], index: int) -> str:
    structured = (problem.get("metrics") or {}).get("structured_problem", {})
    title = structured.get("title") if isinstance(structured, dict) else ""
    if title and title != "\u3010\u9898\u76ee\u3011":
        return str(title)
    match = re.match(r"\s*(\u3010[^\u3011]{1,30}\u3011)", str(problem["ocr_text"]))
    return match.group(1) if match else f"\u9898\u76ee {index}"


def _group_problems(
    problems: list[dict[str, Any]],
) -> OrderedDict[str, OrderedDict[str, list[dict[str, Any]]]]:
    grouped: OrderedDict[str, OrderedDict[str, list[dict[str, Any]]]] = OrderedDict()
    for problem in problems:
        group = str(problem.get("category_group") or "\u672a\u5206\u7c7b")
        category = str(
            problem.get("category_key")
            or problem.get("category")
            or "\u672a\u5206\u7c7b"
        )
        grouped.setdefault(group, OrderedDict()).setdefault(category, []).append(problem)
    return grouped


def _safe_name_part(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"[\x00-\x1f\x7f/\\:*?\"<>|]+", "-", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-. ")
    return normalized or "\u672a\u5206\u7c7b"


def _trim_utf8(value: str, maximum: int) -> str:
    while len(value.encode("utf-8")) > maximum and value:
        value = value[:-1]
    return value.rstrip("-_. ")


def build_export_basename(
    problems: list[dict[str, Any]],
    *,
    generated_at: datetime | None = None,
    random_code: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().astimezone()
    timestamp = generated_at.strftime("%Y%m%d-%H%M")
    counts = Counter(
        str(problem.get("category_group") or "\u672a\u5206\u7c7b")
        for problem in problems
    )
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    categories = [
        f"{_safe_name_part(name)}{count}" for name, count in ordered[:3]
    ]
    remaining = len(ordered) - len(categories)
    if remaining > 0:
        categories.append(f"\u7b49{remaining}\u7c7b")
    category_summary = "-".join(categories) or "\u672a\u5206\u7c7b0"
    code = random_code or "".join(secrets.choice(_CODE_ALPHABET) for _ in range(4))
    if not re.fullmatch(r"[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{4}", code):
        raise ValueError("Invalid export display code")
    category_summary = _trim_utf8(category_summary, 130)
    return f"{timestamp}-{category_summary}-{code}"


def markdown_archive_path(batch_id: str, output_dir: Path) -> Path:
    safe_batch = re.sub(r"[^A-Za-z0-9_-]", "", batch_id)[:8] or "batch"
    return output_dir / f"mistake-book-{safe_batch}-obsidian.zip"


def _resource_path(problem: dict[str, Any], asset: str) -> Path:
    if Path(asset).name != asset:
        raise ValueError("Unsafe compact resource path")
    artifact_dir = Path(str(problem.get("selected_artifact") or "")).resolve().parent
    unresolved = artifact_dir / asset
    candidate = unresolved.resolve()
    if unresolved.is_symlink() or candidate.parent != artifact_dir or not candidate.is_file():
        raise ValueError(
            f"Missing or unsafe compact resource for problem {problem.get('id')}"
        )
    return candidate


def _attachment_name(problem: dict[str, Any], index: int, source: Path) -> str:
    problem_id = re.sub(r"[^A-Za-z0-9_-]", "", str(problem["id"]))
    suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg"} else ".png"
    return f"resource-{problem_id}-{index:02d}{suffix}"


def _block_asset(block: dict[str, Any]) -> str:
    if block.get("type") == "image":
        return str(block.get("asset") or "")
    if (
        block.get("type") == "latex"
        and block.get("recognition_state")
        in {"needs_review", "image_fallback", "human_verified_image"}
    ):
        return str(block.get("clean_crop_asset") or "")
    return ""


def _render_blocks(
    problem: dict[str, Any],
    resource_names: dict[tuple[str, int], str],
) -> list[str]:
    model = content_blocks_for_problem(problem)
    rendered: list[str] = []
    image_index = 0
    first_text = True
    title = _problem_title(problem, 1)
    flow: list[str] = []
    for row in content_block_rows(model):
        pieces: list[str] = []
        for block in row:
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text") or "")
                if first_text:
                    text = re.sub(
                        rf"^\s*{re.escape(title)}\s*",
                        "",
                        text,
                        count=1,
                    )
                    first_text = False
                if text:
                    pieces.append(_escape_markdown(text))
                continue
            if block_type == "latex":
                fallback_asset = _block_asset(block)
                if fallback_asset:
                    image_index += 1
                    name = resource_names[(str(problem["id"]), image_index)]
                    pieces.append(f"![[attachments/{name}]]")
                else:
                    latex = str(block.get("latex") or "").strip()
                    if latex:
                        pieces.append(
                            f"$${latex}$$"
                            if block.get("display")
                            else f"${latex}$"
                        )
                continue
            if block_type == "image":
                image_index += 1
                name = resource_names[(str(problem["id"]), image_index)]
                pieces.append(f"![[attachments/{name}]]")
        line = "".join(pieces)
        if line:
            standalone = (
                len(row) == 1
                and (
                    row[0].get("type") == "image"
                    or bool(row[0].get("display"))
                )
            )
            if standalone:
                if flow:
                    rendered.append("".join(flow))
                    flow = []
                rendered.append(line)
            else:
                flow.append(line)
    if flow:
        rendered.append("".join(flow))
    return rendered


def _render_markdown(
    batch_id: str,
    problems: list[dict[str, Any]],
    resource_names: dict[tuple[str, int], str],
) -> str:
    generated_at = datetime.now().astimezone().isoformat()
    grouped = _group_problems(problems)
    lines = [
        "---",
        "type: olympiad-mistake-book",
        f"batch_id: {json.dumps(batch_id, ensure_ascii=False)}",
        f"created_at: {json.dumps(generated_at)}",
        f"problem_count: {len(problems)}",
        "cssclasses:",
        "  - mistake-book-print",
        "tags:",
        "  - mistake-book",
        "  - olympiad",
        "---",
        "",
        "# \u5c0f\u5965\u9519\u9898\u96c6",
        "",
        '<div class="mb-print-help">\u6253\u5370\u524d\u8bf7\u5728 Obsidian \u4e2d\u542f\u7528 '
        '`mistake-book-print.css` \u6837\u5f0f\u7247\u6bb5\u3002</div>',
        "",
        f"\u5171 {len(problems)} \u9053\u9898\u3002",
        "",
        "## \u76ee\u5f55",
        "",
    ]
    problem_number = 0
    for group, categories in grouped.items():
        group_anchor = _anchor("group", group)
        lines.append(f"- [[#^{group_anchor}|{_wikilink_alias(group)}]]")
        for category, items in categories.items():
            category_anchor = _anchor("category", group, category)
            lines.append(
                f"  - [[#^{category_anchor}|{_wikilink_alias(category)}]]"
            )
            for problem in items:
                problem_number += 1
                title = _problem_title(problem, problem_number)
                problem_anchor = _anchor("problem", str(problem["id"]))
                lines.append(
                    f"    - [[#^{problem_anchor}|{_wikilink_alias(title)}]]"
                )

    problem_number = 0
    for group, categories in grouped.items():
        lines.extend(["", f"## {_escape_markdown(group)} ^{_anchor('group', group)}"])
        for category, items in categories.items():
            lines.extend(
                [
                    "",
                    f"### {_escape_markdown(category)} "
                    f"^{_anchor('category', group, category)}",
                ]
            )
            for problem in items:
                problem_number += 1
                title = _problem_title(problem, problem_number)
                lines.extend(
                    [
                        "",
                        f"#### {_escape_markdown(title)} "
                        f"^{_anchor('problem', str(problem['id']))}",
                        "",
                        *_render_blocks(problem, resource_names),
                        "",
                        '<div class="mb-answer-space mb-normal"></div>',
                    ]
                )
                if problem_number % 2 == 0 and problem_number < len(problems):
                    lines.extend(["", '<div class="mb-page-break"></div>'])
    return "\n".join(lines).rstrip() + "\n"


def export_markdown(
    batch_id: str,
    problems: list[dict[str, Any]],
    output_dir: Path,
    *,
    allow_partial: bool = False,
    generated_at: datetime | None = None,
    random_code: str | None = None,
) -> MarkdownExportResult:
    eligible = [problem for problem in problems if markdown_exportable(problem)]
    ineligible = [problem for problem in problems if not markdown_exportable(problem)]
    if ineligible and not allow_partial:
        raise ValueError(
            f"\u4ecd\u6709 {len(ineligible)} \u9053\u9898\u672a\u5b8c\u6210\u4eba\u5de5\u786e\u8ba4"
            "\u6216\u672a\u9009\u62e9\u767d\u5e95\u91cd\u5efa\u56fe\uff0c\u4e0d\u80fd\u5bfc\u51fa Markdown"
        )
    if not eligible:
        raise ValueError("\u6ca1\u6709\u53ef\u5bfc\u51fa\u7684\u5df2\u786e\u8ba4\u91cd\u5efa\u9898\u76ee")

    basename = build_export_basename(
        eligible,
        generated_at=generated_at,
        random_code=random_code,
    )
    filename = f"{basename}.zip"
    note_name = f"{basename}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=".compact-export-",
        suffix=".tmp",
        dir=output_dir,
        delete=False,
    )
    temporary = Path(temporary_file.name)
    temporary_file.close()
    resource_names: dict[tuple[str, int], str] = {}
    resource_paths: dict[tuple[str, int], Path] = {}
    archive_names: set[str] = set()
    try:
        for problem in eligible:
            image_index = 0
            for block in content_blocks_for_problem(problem).get("blocks", []):
                asset = _block_asset(block)
                if not asset:
                    continue
                image_index += 1
                source = _resource_path(problem, asset)
                name = _attachment_name(problem, image_index, source)
                archive_name = f"attachments/{name}"
                if archive_name in archive_names:
                    raise ValueError("Duplicate attachment path in Markdown archive")
                archive_names.add(archive_name)
                key = (str(problem["id"]), image_index)
                resource_names[key] = name
                resource_paths[key] = source
        markdown = _render_markdown(batch_id, eligible, resource_names)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(note_name, markdown.encode("utf-8"))
            archive.writestr(
                ".obsidian/snippets/mistake-book-print.css",
                _PRINT_CSS.encode("utf-8"),
            )
            for key, source in resource_paths.items():
                archive.writestr(
                    f"attachments/{resource_names[key]}",
                    source.read_bytes(),
                )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return MarkdownExportResult(path=target, filename=filename, note_name=note_name)
