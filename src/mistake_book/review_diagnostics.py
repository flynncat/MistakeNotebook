from __future__ import annotations

from typing import Any


def _diagnostic(reason: str, metrics: dict[str, Any]) -> dict[str, str]:
    structured = metrics.get("structured_problem")
    structured = structured if isinstance(structured, dict) else {}

    if reason == "尚未建立该题的人工真值，禁止自动通过":
        return {
            "severity": "warning",
            "title": "尚无人工标准答案",
            "detail": "当前结果没有经过人工确认，因此系统不会自动标记为通过；这不表示题目内容已识别错误。",
        }

    if reason == "双 OCR 数字或比例不一致":
        primary = structured.get("primary_numbers") or []
        secondary = structured.get("secondary_numbers") or []
        return {
            "severity": "critical",
            "title": "两套 OCR 识别出的数字不一致",
            "detail": f"主 OCR：{primary}；辅助 OCR：{secondary}。数字差异可能改变题意，需要核对原图。",
        }

    if reason.startswith("Formula ") and "requires review" in reason:
        return {
            "severity": "warning",
            "title": "\u516c\u5f0f\u8bc6\u522b\u9700\u8981\u6821\u6b63",
            "detail": "\u8bf7\u5bf9\u7167\u516c\u5f0f\u539f\u56fe\u4fee\u6539 LaTeX\uff0c\u5b9e\u65f6\u9884\u89c8\u901a\u8fc7\u540e\u4fdd\u5b58\u3002",
        }

    if reason.startswith("Formula ") and "no safe LaTeX" in reason:
        return {
            "severity": "warning",
            "title": "\u516c\u5f0f\u5df2\u4fdd\u7559\u4e3a\u539f\u56fe",
            "detail": "\u6a21\u578b\u672a\u751f\u6210\u53ef\u5b89\u5168\u6e32\u67d3\u7684 LaTeX\uff0c\u8bf7\u5728\u516c\u5f0f\u7f16\u8f91\u5668\u4e2d\u624b\u52a8\u6821\u6b63\u3002",
        }

    if reason.startswith("Formula OCR unavailable"):
        return {
            "severity": "warning",
            "title": "\u516c\u5f0f\u8bc6\u522b\u6a21\u578b\u4e0d\u53ef\u7528",
            "detail": "\u5df2\u4fdd\u7559\u539f\u6709\u5185\u5bb9\u5757\uff0c\u8bf7\u8fd0\u884c\u516c\u5f0f\u6a21\u578b\u5b89\u88c5\u811a\u672c\u540e\u91cd\u65b0\u5904\u7406\u3002",
        }

    critical_markers = (
        "未定位到印刷题号",
        "题号格式未可靠恢复",
        "重建题干过短",
        "题干差异过大",
        "图",
        "结构",
        "拓扑",
        "数字",
        "比例",
    )
    if any(marker in reason for marker in critical_markers):
        return {
            "severity": "critical",
            "title": reason,
            "detail": "该问题可能改变题目内容或条件，需要对照原图核查。",
        }

    return {
        "severity": "warning",
        "title": reason,
        "detail": "该项会阻止系统自动通过，但尚不能据此判断重建内容存在错误。",
    }


def build_review_diagnostics(
    metrics: dict[str, Any] | None,
    error: str | None = None,
) -> list[dict[str, str]]:
    values = metrics if isinstance(metrics, dict) else {}
    reasons = values.get("review_reasons")
    reason_list = reasons if isinstance(reasons, list) else []
    diagnostics = [
        _diagnostic(str(reason), values)
        for reason in reason_list
        if str(reason).strip() and reason != "第二 OCR 未识别到完整题干"
    ]
    if error:
        diagnostics.append(
            {
                "severity": "critical",
                "title": "处理失败",
                "detail": str(error),
            }
        )
    return diagnostics
