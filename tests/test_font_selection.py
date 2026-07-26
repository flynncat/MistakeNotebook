from __future__ import annotations

import pytest

from mistake_book import font_selection


def test_raster_input_prefers_verified_microsoft_yahei() -> None:
    font, profile = font_selection.load_print_font(32)
    assert font.getmask("错题").getbbox() is not None
    assert profile.detected_family is None
    assert profile.rendered_family == "微软雅黑"
    assert profile.confidence == 0
    assert profile.source == "raster_no_font_metadata"


def test_missing_chinese_sans_font_fails_clearly(monkeypatch) -> None:
    monkeypatch.setattr(font_selection, "_REGULAR_CANDIDATES", ())
    with pytest.raises(RuntimeError, match="缺少支持中文"):
        font_selection.load_print_font(24)
