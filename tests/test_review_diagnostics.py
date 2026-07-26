from mistake_book.review_diagnostics import build_review_diagnostics


def test_secondary_ocr_failure_is_not_shown_as_actionable_diagnostic() -> None:
    diagnostics = build_review_diagnostics(
        {
            "review_reasons": ["第二 OCR 未识别到完整题干"],
            "structured_problem": {
                "title": "【例题8】",
                "secondary_raw_text": "(Gi/zi8) 件玩具售优22元",
            },
        }
    )
    assert diagnostics == []


def test_grid_ocr_failure_is_kept_out_of_user_review_items() -> None:
    diagnostics = build_review_diagnostics(
        {
            "review_reasons": ["第二 OCR 未识别到完整题干"],
            "structured_problem": {
                "title": "【例题5】",
                "figure": "grid_2x10",
                "secondary_raw_text": "PileQ5) 第第第列列列 se—aAPBAY",
            },
        }
    )
    assert diagnostics == []


def test_number_disagreement_is_critical_and_shows_values() -> None:
    diagnostics = build_review_diagnostics(
        {
            "review_reasons": ["双 OCR 数字或比例不一致"],
            "structured_problem": {
                "primary_numbers": ["22", "37:25"],
                "secondary_numbers": ["22", "37:2"],
            },
        }
    )
    assert diagnostics[0]["severity"] == "critical"
    assert "37:25" in diagnostics[0]["detail"]
    assert "37:2" in diagnostics[0]["detail"]


def test_missing_ground_truth_is_warning_not_content_error() -> None:
    diagnostics = build_review_diagnostics(
        {"review_reasons": ["尚未建立该题的人工真值，禁止自动通过"]}
    )
    assert diagnostics[0]["severity"] == "warning"
    assert "不表示题目内容已识别错误" in diagnostics[0]["detail"]
