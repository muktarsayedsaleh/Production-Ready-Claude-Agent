import asyncio

import pytest

from agents.research.tools import CalculatorTool

calculator = CalculatorTool()


def run(expression: object) -> str:
    return asyncio.run(calculator.run({"expression": expression}))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("7 - 10", "-3"),
        ("6 * 7", "42"),
        ("7 / 2", "3.5"),
        ("2 ** 10", "1024"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2 ** 3 ** 2", "512"),
        ("-(3 - 5) * +2", "4"),
        ("1.5 * 4", "6.0"),
        ("2 ** -1", "0.5"),
        ("  ((1 + 2) * (3 + 4)) / 7  ", "3.0"),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert run(expression) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('id')", "Unsupported syntax"),
        ("abs(-1)", "Unsupported syntax"),
        ("x + 1", "Unsupported syntax"),
        ("(1).real", "Unsupported syntax"),
        ("7 // 2", "Unsupported syntax"),
        ("7 % 2", "Unsupported syntax"),
        ("True + 1", "Unsupported syntax"),
        ("'a' * 3", "Unsupported syntax"),
        ("[1, 2]", "Unsupported syntax"),
        ("1 +", "Invalid expression"),
        ("1 / 0", "Division by zero"),
        ("0 ** -1", "Division by zero"),
        ("9 ** 9 ** 9", "exceed 1000 digits"),
        ("10.0 ** 400", "Result is too large"),
    ],
)
def test_rejects_unsafe_or_invalid_input(expression: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run(expression)


def test_rejects_overlong_and_non_string_expressions() -> None:
    with pytest.raises(ValueError, match="longer than 500"):
        run("1+" * 300 + "1")
    with pytest.raises(ValueError, match="non-empty string"):
        run(42)
