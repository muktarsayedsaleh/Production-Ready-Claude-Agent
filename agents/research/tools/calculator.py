import ast
import math
import operator
from collections.abc import Callable
from typing import Any, ClassVar, cast

Number = int | float

_BINARY_OPS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool:
    """Evaluates arithmetic by walking a parsed AST; nothing is ever passed to eval()."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression with numbers, + - * / ** and parentheses, "
        "e.g. '(2 + 3) * 4 ** 2 / 10'. Returns the result as text."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The expression to evaluate."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    def __init__(self, max_length: int = 500, max_result_digits: int = 1000) -> None:
        self._max_length = max_length
        self._max_result_digits = max_result_digits

    async def run(self, args: dict[str, Any]) -> str:
        expression = args.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("'expression' must be a non-empty string")
        return str(self.evaluate(expression))

    def evaluate(self, expression: str) -> Number:
        if len(expression) > self._max_length:
            raise ValueError(f"Expression is longer than {self._max_length} characters")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError:
            raise ValueError(f"Invalid expression: {expression!r}") from None
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return cast(Number, node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow):
                self._check_power(left, right)
            try:
                return _BINARY_OPS[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero") from None
            except OverflowError:
                raise ValueError("Result is too large") from None
        raise ValueError(f"Unsupported syntax: {ast.unparse(node)!r}")

    def _check_power(self, base: Number, exponent: Number) -> None:
        # Refuse powers whose result would be enormous (e.g. 9 ** 9 ** 9) before computing them.
        if abs(base) > 1 and exponent > 0:
            digits = exponent * math.log10(abs(base))
            if digits > self._max_result_digits:
                raise ValueError(f"Result would exceed {self._max_result_digits} digits")
        if isinstance(base, int) and base == 0 and exponent < 0:
            raise ValueError("Division by zero")
