import operator


def if_then_else(input, output1, output2):
    """
    Returns one of two outputs based on the truth value of the input.
    Args:
        input (bool): Condition to evaluate. If True, output1 is returned; otherwise, output2 is returned.
        output1 (float): Value to return if input is True.
        output2 (float): Value to return if input is False.
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
        left (float): The numerator.
        right (float): The denominator.
    Returns:
        float: The result of the division, or 1 if `right` is zero.
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
    
PRIMITIVES = [
    {"id": "if", "name": "If Then Else", "type": "Float", "function": if_then_else},
    {"id": "add", "name": "Addition", "type": "Float", "function": operator.add},
    {"id": "sub", "name": "Substraction", "type": "Float", "function": operator.sub},
    {"id": "mul", "name": "Multiplication", "type": "Float", "function": operator.mul},
    {
        "id": "div",
        "name": "Protected Division",
        "type": "Float",
        "function": protected_div,
    },
    {"id": "and", "name": "And", "type": "Boolean", "function": operator.and_},
    {"id": "or", "name": "Or", "type": "Boolean", "function": operator.or_},
    {"id": "not", "name": "Not", "type": "Boolean", "function": operator.not_},
    {"id": "lt", "name": "Lower Than", "type": "Boolean", "function": operator.lt},
    {"id": "eq", "name": "Equal", "type": "Boolean", "function": operator.eq},
]