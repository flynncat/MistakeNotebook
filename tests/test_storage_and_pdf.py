from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from PIL import Image

from mistake_book.pdf_export import _group_problems, export_pdf
from mistake_book.storage import Storage, content_fingerprint


def test_storage_round_trip(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "data")
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source)

    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "题目.png", source)
    storage.update_problem(
        problem_id,
        status="ready",
        category_group="计数",
        category="染色问题",
        category_key="染色问题",
        summary="计算染色方案",
        category_confidence=0.91,
        ocr_text="共有多少种不同的方法？",
        metrics_json={"score": 0.9},
    )

    problem = storage.get_problem(problem_id)
    assert problem is not None
    assert problem["filename"] == "题目.png"
    assert problem["category_group"] == "计数"
    assert problem["category_key"] == "染色问题"
    assert problem["metrics"]["score"] == 0.9
    newest = storage.list_problems(
        category_group="计数",
        category="染色问题",
        query="多少种",
    )
    assert newest["total"] == 1
    assert newest["items"][0]["id"] == problem_id


def test_asset_fingerprint_normalizes_text_but_keeps_figure_kind() -> None:
    first = content_fingerprint("【例题 8】 共有多少种？", "none")
    second = content_fingerprint("[例题8]共有多少种?", "none")
    with_figure = content_fingerprint("[例题8]共有多少种?", "grid_2x10")
    assert first == second
    assert first != with_figure


def test_exact_source_can_be_reprocessed_and_relative_path_is_kept(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "data")
    source = tmp_path / "source.png"
    Image.new("RGB", (30, 20), "white").save(source)
    first_batch = storage.create_batch()
    first = storage.add_problem(
        first_batch,
        "source.png",
        source,
        source_relative_path="课本/第一章/source.png",
    )
    second_batch = storage.create_batch()
    second = storage.add_problem(second_batch, "copy.png", source)
    assert first is not None
    assert second is not None
    problem = storage.get_problem(first)
    repeated = storage.get_problem(second)
    assert problem is not None
    assert repeated is not None
    assert problem["source_relative_path"] == "课本/第一章/source.png"
    assert len(problem["source_sha256"]) == 64
    assert repeated["source_sha256"] == problem["source_sha256"]


def test_publishing_latest_asset_deletes_older_record_and_files(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "data")
    ids: list[str] = []
    batch_ids: list[str] = []
    for index, color in enumerate(("white", "gray")):
        source = tmp_path / f"source-{index}.png"
        Image.new("RGB", (30, 20), color).save(source)
        batch_id = storage.create_batch()
        batch_ids.append(batch_id)
        problem_id = storage.add_problem(batch_id, source.name, source)
        assert problem_id is not None
        artifact_dir = storage.files_dir / batch_id / problem_id
        artifact_dir.mkdir(parents=True)
        artifact = artifact_dir / "question.png"
        Image.new("RGB", (40, 30), "white").save(artifact)
        storage.update_problem(
            problem_id,
            status="ready",
            review_status="accepted",
            selected_artifact=str(artifact),
            ocr_text="【例题8】共有多少种方法？",
            metrics_json={"structured_problem": {"figure": "none"}},
        )
        result = storage.publish_asset(problem_id, selected_kind="reconstructed")
        ids.append(problem_id)

    assert result["replaced_count"] == 1
    assert storage.get_problem(ids[0]) is None
    assert storage.get_problem(ids[1]) is not None
    assert not (storage.files_dir / batch_ids[0] / ids[0]).exists()
    assets = storage.list_assets()
    assert assets["total"] == 1
    assert assets["items"][0]["id"] == ids[1]


def test_storage_migrates_legacy_category_without_changing_display_text(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    database = data_dir / "mistake_book.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.executescript(
            """
            CREATE TABLE batches (
                id TEXT PRIMARY KEY,status TEXT,created_at TEXT,updated_at TEXT,pdf_path TEXT
            );
            CREATE TABLE problems (
                id TEXT PRIMARY KEY,batch_id TEXT,filename TEXT,source_path TEXT,status TEXT,
                review_status TEXT,selected_artifact TEXT,category TEXT,ocr_text TEXT,
                confidence REAL,metrics_json TEXT,error TEXT,created_at TEXT,updated_at TEXT
            );
            CREATE TABLE categories (
                id TEXT PRIMARY KEY,name TEXT UNIQUE,description TEXT,aliases_json TEXT,created_at TEXT
            );
            INSERT INTO batches VALUES ('b','ready','2026-01-01','2026-01-01',NULL);
            INSERT INTO problems VALUES (
                'p','b','old.png','/tmp/old.png','ready','not_required',NULL,
                '计数·数位进位','【例题7】题干',0.8,
                '{"recognition_summary":"进位知识点","review_reasons":["未定位到印刷题号"]}',
                NULL,
                '2026-01-01','2026-01-01'
            );
            """
        )

    storage = Storage(data_dir)
    problem = storage.get_problem("p")
    assert problem is not None
    assert problem["category"] == "计数·数位进位"
    assert problem["category_group"] == "计数"
    assert problem["category_key"] == "数位进位"
    assert problem["summary"] == "进位知识点"
    assert problem["category_source"] == "migrated"
    assert problem["metrics"]["review_reasons"] == ["第二 OCR 未识别到完整题干"]
    assert (data_dir / "mistake_book.pre-assets.sqlite3").exists()


def test_storage_removes_conflicts_from_unreliable_secondary_ocr(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    storage = Storage(data_dir)
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "题目.png", source)
    storage.update_problem(
        problem_id,
        metrics_json={
            "review_reasons": [
                "双 OCR 题干差异过大",
                "双 OCR 数字或比例不一致",
                "尚未建立该题的人工真值，禁止自动通过",
            ],
            "structured_problem": {
                "text_similarity": 0.41,
                "review_reasons": [
                    "双 OCR 题干差异过大",
                    "双 OCR 数字或比例不一致",
                ],
            },
        },
    )

    migrated = Storage(data_dir).get_problem(problem_id)
    assert migrated is not None
    assert migrated["metrics"]["review_reasons"] == [
        "尚未建立该题的人工真值，禁止自动通过"
    ]
    assert migrated["metrics"]["structured_problem"]["review_reasons"] == []


def test_storage_removes_obsolete_tesseract_path_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    storage = Storage(data_dir)
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source)
    batch_id = storage.create_batch()
    problem_id = storage.add_problem(batch_id, "题目.png", source)
    storage.update_problem(
        problem_id,
        metrics_json={
            "review_reasons": [
                "未安装 Tesseract",
                "尚未建立该题的人工真值，禁止自动通过",
            ]
        },
    )

    migrated = Storage(data_dir).get_problem(problem_id)
    assert migrated is not None
    assert migrated["metrics"]["review_reasons"] == [
        "尚未建立该题的人工真值，禁止自动通过"
    ]


def test_pdf_is_a4_and_requires_review_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "cleaned.png"
    Image.new("RGB", (1200, 500), "white").save(image_path)
    ready = {
        "id": "p1",
        "status": "ready",
        "review_status": "not_required",
        "selected_artifact": str(image_path),
        "category_group": "计数",
        "category_key": "计数综合",
        "category": "计数问题",
        "ocr_text": "共有多少种？",
    }
    pending = {
        **ready,
        "id": "p2",
        "status": "needs_review",
        "review_status": "pending",
    }

    try:
        export_pdf("batch", [ready, pending], tmp_path)
    except ValueError as error:
        assert "未通过确认" in str(error)
    else:
        raise AssertionError("有待确认题目时不应生成最终 PDF")

    target = export_pdf("batch", [ready, pending], tmp_path, allow_partial=True)
    assert target.exists()
    assert target.read_bytes().startswith(b"%PDF")


def test_pdf_groups_toc_by_domain_then_category() -> None:
    grouped = _group_problems(
        [
            {"category_group": "计数", "category_key": "数位进位", "category": "旧值"},
            {"category_group": "应用", "category_key": "比例问题", "category": "旧值"},
            {"category_group": "计数", "category_key": "染色问题", "category": "旧值"},
            {"category_group": "计数", "category_key": "染色问题", "category": "旧值"},
        ]
    )
    assert list(grouped) == ["计数", "应用"]
    assert list(grouped["计数"]) == ["数位进位", "染色问题"]
    assert len(grouped["计数"]["染色问题"]) == 2


def test_pdf_places_at_most_two_problems_on_each_content_page(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "short-question.png"
    Image.new("RGB", (1200, 100), "white").save(image_path)
    problems = [
        {
            "id": f"p{index}",
            "status": "ready",
            "review_status": "accepted",
            "selected_artifact": str(image_path),
            "category_group": "计数",
            "category_key": "计数综合",
            "category": "计数综合",
            "ocr_text": f"第{index}题",
        }
        for index in range(5)
    ]
    target = export_pdf("two-per-page", problems, tmp_path / "exports")
    page_count = len(re.findall(rb"/Type\s*/Page(?!s)", target.read_bytes()))
    assert page_count == 4  # 1 页目录 + 3 页正文
