from __future__ import annotations

import re
from dataclasses import dataclass


TAXONOMY: dict[str, tuple[str, ...]] = {
    "计数": ("染色问题", "数位进位", "组数问题", "排列问题", "组合计数", "计数综合"),
    "组合": ("抽屉原理", "容斥原理"),
    "数论": ("整除余数", "质因数", "数论综合"),
    "几何": ("面积周长", "图形计数", "几何综合"),
    "应用": ("比例问题", "工程问题", "浓度利润", "应用题"),
    "行程": ("相遇追及", "行程综合"),
    "逻辑": ("逻辑推理",),
    "未分类": ("未分类",),
}


@dataclass(frozen=True)
class Classification:
    group: str
    category: str
    summary: str
    confidence: float
    source: str
    review_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Rule:
    group: str
    category: str
    summary: str
    terms: tuple[tuple[str, int], ...]

    def score(self, text: str) -> int:
        return sum(weight for pattern, weight in self.terms if re.search(pattern, text))


_RULES = (
    _Rule(
        "组合",
        "抽屉原理",
        "利用抽屉原理保证至少若干对象具有相同性质",
        ((r"至少", 2), (r"完全相同|相同", 4), (r"列", 2), (r"2[×xX*]10", 1)),
    ),
    _Rule(
        "计数",
        "染色问题",
        "利用分类、乘法原理或容斥计算染色方案",
        ((r"染色", 5), (r"涂色", 5), (r"颜色", 2), (r"彩笔", 1)),
    ),
    _Rule(
        "计数",
        "数位进位",
        "分析数位相加与进位条件",
        ((r"进位", 5), (r"相加", 1), (r"三位数|四位数", 1)),
    ),
    _Rule(
        "计数",
        "组数问题",
        "按数位限制组成整数并计数",
        ((r"组成", 3), (r"三位数|四位数|多位数", 3), (r"卡片.*数字|数字.*卡片", 2)),
    ),
    _Rule(
        "应用",
        "比例问题",
        "根据前后数量之比建立关系",
        ((r"之比|钱数比", 5), (r"钱数", 2), (r"售价|买", 1)),
    ),
    _Rule(
        "计数",
        "排列问题",
        "研究有顺序的排列方案",
        ((r"排列|排成|站队|顺序", 4), (r"多少种|共有多少", 2)),
    ),
    _Rule(
        "计数",
        "组合计数",
        "研究无顺序的选择方案",
        ((r"组合|选出|取出|选法", 4), (r"多少种|共有多少", 2)),
    ),
    _Rule(
        "组合",
        "容斥原理",
        "用容斥原理处理重复计数",
        ((r"至少一个|都不|重复计算", 3), (r"容斥", 5)),
    ),
    _Rule(
        "数论",
        "整除余数",
        "研究整除、倍数和余数性质",
        ((r"整除|余数", 5), (r"倍数|约数|因数", 3)),
    ),
    _Rule(
        "数论",
        "质因数",
        "研究质数、合数和质因数分解",
        ((r"质数|合数|质因数", 5),),
    ),
    _Rule(
        "几何",
        "面积周长",
        "计算或比较图形面积与周长",
        ((r"面积|周长", 5), (r"三角形|正方形|长方形|圆", 1)),
    ),
    _Rule(
        "几何",
        "图形计数",
        "按结构分类统计图形数量",
        ((r"多少个.*(?:三角形|正方形|长方形|图形)", 5),),
    ),
    _Rule(
        "行程",
        "相遇追及",
        "根据路程、速度和时间分析相遇或追及",
        ((r"相遇|追及", 5), (r"路程|速度|时间", 2)),
    ),
    _Rule(
        "行程",
        "行程综合",
        "根据路程、速度和时间关系求解",
        ((r"路程|速度", 3), (r"时间", 2)),
    ),
    _Rule(
        "应用",
        "工程问题",
        "利用工作总量和效率关系求解",
        ((r"工程|工作效率|完成.*工作", 5),),
    ),
    _Rule(
        "应用",
        "浓度利润",
        "分析浓度、成本、售价或利润关系",
        ((r"浓度|利润|成本", 5),),
    ),
    _Rule(
        "逻辑",
        "逻辑推理",
        "根据条件进行真假与顺序推理",
        ((r"一定|可能|真假|推理", 3), (r"条件", 1)),
    ),
    _Rule(
        "计数",
        "计数综合",
        "使用分类计数或乘法原理",
        ((r"共有多少|有多少种|多少种不同", 2), (r"方法|方案|种", 2)),
    ),
    _Rule(
        "应用",
        "应用题",
        "建立数量关系解决实际问题",
        ((r"鸡兔同笼|一共.*还剩|售价", 3),),
    ),
)


def classify_by_rules(
    text: str,
    taxonomy: dict[str, tuple[str, ...]] | None = None,
) -> Classification:
    compact = re.sub(r"\s+", "", text)
    allowed = taxonomy or TAXONOMY
    candidates = [
        rule for rule in _RULES if rule.category in allowed.get(rule.group, ())
    ]
    if not candidates:
        return Classification(
            group="未分类",
            category="未分类",
            summary="当前没有可用的自动分类",
            confidence=0.0,
            source="rules",
            review_reasons=("需要人工选择题型",),
        )
    ranked = sorted(
        ((rule.score(compact), index, rule) for index, rule in enumerate(candidates)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, _, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0
    if best_score < 3:
        return Classification(
            group="未分类",
            category="未分类",
            summary="暂未识别出稳定的小奥知识点",
            confidence=0.0,
            source="rules",
            review_reasons=("未识别出稳定题型",),
        )
    confidence = min(
        0.98,
        0.55 + 0.08 * (best_score - 3) + 0.04 * (best_score - second_score),
    )
    reasons = ("分类置信度较低",) if confidence < 0.75 else ()
    return Classification(
        group=best.group,
        category=best.category,
        summary=best.summary,
        confidence=round(confidence, 4),
        source="rules",
        review_reasons=reasons,
    )


def valid_category(
    group: str,
    category: str,
    taxonomy: dict[str, tuple[str, ...]] | None = None,
) -> bool:
    return category in (taxonomy or TAXONOMY).get(group, ())


def taxonomy_payload(
    taxonomy: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, object]]:
    selected = taxonomy or TAXONOMY
    return [
        {"name": group, "categories": list(categories)}
        for group, categories in selected.items()
        if group != "未分类"
    ]


def legacy_category_key(category: str | None) -> tuple[str, str]:
    value = (category or "").strip()
    if not value:
        return "未分类", "未分类"
    if value.startswith("计数·"):
        key = value.split("·", 1)[1]
        return "计数", key if key in TAXONOMY["计数"] else "计数综合"
    mappings = {
        "抽屉原理": ("组合", "抽屉原理"),
        "比例问题": ("应用", "比例问题"),
        "应用题": ("应用", "应用题"),
        "数论问题": ("数论", "数论综合"),
        "几何问题": ("几何", "几何综合"),
        "行程问题": ("行程", "行程综合"),
        "计数问题": ("计数", "计数综合"),
    }
    return mappings.get(value, ("未分类", "未分类"))
