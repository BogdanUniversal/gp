import operator


def if_then_else(input, output1, output2):
    """
    Returns one of two outputs based on the truth value of the input.
    Args:
        input (bool): Condition to evaluate. If True, output1 is returned; otherwise, output2 is returned.
        output1 (Primitive): Value to return if input is True.
        output2 (Primitive): Value to return if input is False.
    Returns:
        Any: output1 if input is True, otherwise output2.
    """
    if input:
        return output1
    else:
        return output2
    
    
def protected_div(left, right):
    """
    Safely performs division between two numbers.
    Divides `left` by `right` and returns the result. If a division by zero occurs,
    returns 1 instead of raising an exception.
    Args:
        left (Primitive): The numerator.
        right (Primitive): The denominator.
    Returns:
        Primitive: The result of the division, or 1 if `right` is zero.
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
    
PRIMITIVES = [
    {"id": "if", "name": "If Then Else", "type": "Primitive", "function": if_then_else, "in": [bool, float, float], "out": float},
    {"id": "add", "name": "Addition", "type": "Primitive", "function": operator.add, "in": [float, float], "out": float},
    {"id": "sub", "name": "Substraction", "type": "Primitive", "function": operator.sub, "in": [float, float], "out": float},
    {"id": "mul", "name": "Multiplication", "type": "Primitive", "function": operator.mul, "in": [float, float], "out": float},
    {
        "id": "div",
        "name": "Protected Division",
        "type": "Primitive",
        "function": protected_div,
        "in": [float, float],
        "out": float,
    },
    {"id": "and", "name": "And", "type": "Primitive", "function": operator.and_, "in": [bool, bool], "out": bool},
    {"id": "or", "name": "Or", "type": "Primitive", "function": operator.or_, "in": [bool, bool], "out": bool},
    {"id": "not", "name": "Not", "type": "Primitive", "function": operator.not_, "in": [bool], "out": bool},
    {"id": "lt", "name": "Lower Than", "type": "Primitive", "function": operator.lt, "in": [float, float], "out": bool},
    {"id": "le", "name": "Lower Equal", "type": "Primitive", "function": operator.le, "in": [float, float], "out": bool},
    {"id": "eq", "name": "Equal", "type": "Primitive", "function": operator.eq, "in": [float, float], "out": bool},
]