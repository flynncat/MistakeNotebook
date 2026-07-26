from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import ImageFont


@dataclass(frozen=True)
class FontProfile:
    detected_family: str | None
    rendered_family: str
    confidence: float
    source: str
    fallback_reason: str
    path: str
    index: int

    def to_dict(self) -> dict[str, str | float | int | None]:
        return asdict(self)


_USER_FONT_DIR = Path.home() / "Library" / "Fonts"
_REGULAR_CANDIDATES = (
    (_USER_FONT_DIR / "msyh.ttc", 0, "微软雅黑"),
    (Path("/Library/Fonts/msyh.ttc"), 0, "微软雅黑"),
    (Path("/System/Library/Fonts/STHeiti Light.ttc"), 1, "黑体"),
    (Path("/System/Library/Fonts/STHeiti Light.ttc"), 0, "黑体"),
    (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), 0, "Noto Sans CJK"),
    (Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"), 0, "Noto Sans CJK"),
)
_BOLD_CANDIDATES = (
    (_USER_FONT_DIR / "msyhbd.ttc", 0, "微软雅黑"),
    (Path("/Library/Fonts/msyhbd.ttc"), 0, "微软雅黑"),
    (Path("/System/Library/Fonts/STHeiti Medium.ttc"), 1, "黑体"),
    (Path("/System/Library/Fonts/STHeiti Medium.ttc"), 0, "黑体"),
    *_REGULAR_CANDIDATES,
)


def _supports_chinese(font: ImageFont.FreeTypeFont) -> bool:
    signatures: set[tuple[tuple[int, int], str]] = set()
    for character in "错题数学":
        mask = font.getmask(character)
        payload = bytes(mask)
        if not payload or not any(payload):
            return False
        signatures.add((mask.size, hashlib.sha1(payload).hexdigest()))
    return len(signatures) >= 3


def load_print_font(
    size: int,
    *,
    bold: bool = False,
) -> tuple[ImageFont.FreeTypeFont, FontProfile]:
    candidates = _BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES
    for path, index, family in candidates:
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size=size, index=index)
        except OSError:
            continue
        if not _supports_chinese(font):
            continue
        return font, FontProfile(
            detected_family=None,
            rendered_family=family,
            confidence=0.0,
            source="raster_no_font_metadata",
            fallback_reason="照片不包含可验证的字体元数据，使用默认微软雅黑",
            path=str(path),
            index=index,
        )
    raise RuntimeError("缺少支持中文的默认字体")


def default_font_metrics() -> dict[str, str | float | int | None]:
    _, profile = load_print_font(24)
    return profile.to_dict()
