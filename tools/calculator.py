import ast
import operator

from langchain_core.tools import tool


MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate(node):
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so exclude it explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        left = _calculate(node.left)
        right = _calculate(node.right)

        # Guard against expressions like 9**9**9**9 that would hang the server.
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_EXPONENT}).")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(_calculate(node.operand))

    raise ValueError("Invalid mathematical expression.")


def evaluate(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return the result as text."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "Calculation error: expression too long."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_calculate(tree.body))
    except ZeroDivisionError:
        return "Calculation error: division by zero."
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Args:
        expression: A plain arithmetic expression using numbers and the
            operators + - * / ** %, for example "25 * 48" or "(3 + 4) ** 2".
            Do not include words or units.
    """
    return evaluate(expression)
