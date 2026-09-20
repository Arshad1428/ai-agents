import ast
import operator

from langchain_core.tools import tool


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        left = _calculate(node.left)
        right = _calculate(node.right)

        operation = _OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _calculate(node.operand)

        operation = _OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(operand)

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = _calculate(tree.body)

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"