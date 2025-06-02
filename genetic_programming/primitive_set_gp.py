import operator


def if_then_else(arg0: bool, arg1: float, arg2: float):
    """
    Returns one of two outputs based on the truth value of the input.
    Args:
        arg0 (bool): Condition to evaluate. If True, arg1 is returned; otherwise, arg2 is returned.
        arg1 (Primitive): Value to return if input is True.
        arg2 (Primitive): Value to return if input is False.
    Returns:
        Any: output1 if input is True, otherwise arg2.
    """
    if arg0:
        return arg1
    else:
        return arg1
    
    
def protected_div(arg0: float, arg1: float):
    """
    Safely performs division between two numbers.
    Divides `arg0` by `arg1` and returns the result. If a division by zero occurs,
    returns 1 instead of raising an exception.
    Args:
        arg0 (Primitive): The numerator.
        arg1 (Primitive): The denominator.
    Returns:
        Primitive: The result of the division, or 1 if `arg1` is zero.
    """
    try:
        return arg0 / arg1
    except ZeroDivisionError:
        return 1
    
    
PRIMITIVES = [
    {"id": "if_then_else", "name": "If Then Else", "type": "Primitive", "function": if_then_else, "in": [bool, float, float], "out": float},
    {"id": "add", "name": "Addition", "type": "Primitive", "function": operator.add, "in": [float, float], "out": float},
    {"id": "sub", "name": "Substraction", "type": "Primitive", "function": operator.sub, "in": [float, float], "out": float},
    {"id": "mul", "name": "Multiplication", "type": "Primitive", "function": operator.mul, "in": [float, float], "out": float},
    {
        "id": "protected_div",
        "name": "Protected Division",
        "type": "Primitive",
        "function": protected_div,
        "in": [float, float],
        "out": float,
    },
    {"id": "and_", "name": "And", "type": "Primitive", "function": operator.and_, "in": [bool, bool], "out": bool},
    {"id": "or_", "name": "Or", "type": "Primitive", "function": operator.or_, "in": [bool, bool], "out": bool},
    {"id": "not_", "name": "Not", "type": "Primitive", "function": operator.not_, "in": [bool], "out": bool},
    {"id": "lt", "name": "Lower Than", "type": "Primitive", "function": operator.lt, "in": [float, float], "out": bool},
    {"id": "le", "name": "Lower Equal", "type": "Primitive", "function": operator.le, "in": [float, float], "out": bool},
    {"id": "eq", "name": "Equal", "type": "Primitive", "function": operator.eq, "in": [float, float], "out": bool},
]