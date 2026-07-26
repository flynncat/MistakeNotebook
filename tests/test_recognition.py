from mistake_book.classification import classify_by_rules
from mistake_book.recognition import _select_punctuation_candidate
from mistake_book.v2_pipeline import _ocr_candidate_score


def test_specific_olympiad_categories_take_priority() -> None:
    cases = {
        "与456相加至少发生一次进位": ("计数", "数位进位"),
        "三个圆的七个区域有多少种染色方法": ("计数", "染色问题"),
        "至少有多少列的颜色完全相同": ("组合", "抽屉原理"),
        "原来钱数之比是37比25": ("应用", "比例问题"),
    }
    for text, expected in cases.items():
        result = classify_by_rules(text)
        assert (result.group, result.category) == expected
        assert result.confidence >= 0.75


def test_unstable_category_requires_review() -> None:
    result = classify_by_rules("求下面问题的结果")
    assert result.category == "未分类"
    assert result.confidence == 0
    assert result.review_reasons


def test_ocr_prefers_source_supported_ellipsis_candidate() -> None:
    line = {
        "text": "13\u3001 \u2022\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
        "confidence": 0.82,
        "alternatives": [
            {
                "text": "13\u3001 \u2022\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
                "confidence": 0.82,
            },
            {
                "text": "13\u3001\u2026\u2026\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
                "confidence": 0.80,
            },
        ],
    }

    assert _select_punctuation_candidate(line) == (
        "13\u3001\u2026\u2026\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f"
    )


def test_ocr_normalizes_direct_dotted_ellipsis_evidence() -> None:
    line = {
        "text": "13\u3001 ...\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
        "confidence": 0.82,
        "alternatives": [],
    }

    assert _select_punctuation_candidate(line) == (
        "13\u3001 \u2026\u2026\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f"
    )


def test_ocr_does_not_guess_ellipsis_without_matching_source_candidate() -> None:
    line = {
        "text": "13\u3001 \u00b7\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
        "confidence": 0.82,
        "alternatives": [
            {
                "text": "14\u3001\u2026\u2026\uff0c\u90a3\u4e483012\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f",
                "confidence": 0.81,
            }
        ],
    }

    assert _select_punctuation_candidate(line) == line["text"]


def test_v2_prefers_ocr_candidate_with_source_ellipsis_evidence() -> None:
    bullet = (
        "1\u30012\u30013\u3001\u2022\uff0c\u90a3\u4e483012"
        "\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f"
    )
    ellipsis = (
        "1\u30012\u30013\u3001\u2026\u2026\uff0c\u90a3\u4e483012"
        "\u662f\u7b2c\u51e0\u4e2a\u6570\uff1f"
    )

    assert _ocr_candidate_score(ellipsis) > _ocr_candidate_score(bullet)
