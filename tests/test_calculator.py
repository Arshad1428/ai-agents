import pytest

from tools.calculator import calculator, evaluate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("25 * 48", "1200"),
        ("(3 + 4) ** 2", "49"),
        ("10 / 4", "2.5"),
        ("-5 + 2", "-3"),
        ("17 % 5", "2"),
    ],
)
def test_basic_math(expression, expected):
    assert evaluate(expression) == expected


def test_division_by_zero():
    assert "division by zero" in evaluate("1 / 0")


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "abs(-1)", "'a' * 3", "True + 1", "x + 1"],
)
def test_rejects_non_arithmetic(expression):
    assert evaluate(expression).startswith("Calculation error")


def test_huge_exponent_is_rejected_quickly():
    assert "Exponent too large" in evaluate("9 ** 9 ** 9 ** 9")


def test_tool_wrapper():
    assert calculator.invoke({"expression": "25 * 48"}) == "1200"
