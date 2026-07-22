from mistake_book.recognition import _rule_category


def test_specific_olympiad_categories_take_priority() -> None:
    assert _rule_category("与456相加至少发生一次进位")[0] == "计数·数位进位"
    assert _rule_category("三个圆的七个区域有多少种染色方法")[0] == "计数·染色问题"
    assert _rule_category("至少有多少列的颜色完全相同")[0] == "抽屉原理"
    assert _rule_category("原来钱数之比是37比25")[0] == "比例问题"
