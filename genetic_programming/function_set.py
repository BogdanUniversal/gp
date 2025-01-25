# from helper import evaluate
import numpy as np
import uuid


def listFunctionsGeneral():
    return [
        {"function": forLoop, "returnType": "body", "parametersTypes": ["body", int, int, int]},
        {"function": forLoop, "returnType": "body", "parametersTypes": ["body", int, int, int]},
        {"function": "body", "returnType": "body", "parametersTypes": None},
        # "ifElse": {"function": ifElse, "returnType": "body", "parametersTypes": []},
    ]
    
    
def listFunctions1D():
    return [
        {"type": "function", "function": swap1D, "returnType": list, "parametersTypes": [list, int, int]},
        {"type": "function", "function": sort1D, "returnType": list, "parametersTypes": [list, bool]},
        {"type": "function", "function": flip1D, "returnType": list, "parametersTypes": [list]},
    ]


def listFunctionsInt():
    return [
        {"type": "function", "function": length1D, "returnType": int, "parametersTypes": [list]},
        {"type": "function", "function": convertInt, "returnType": int, "parametersTypes": [int]},
        {"type": "function", "function": absolute, "returnType": int, "parametersTypes": [int]},
        {"type": "function", "function": getMax, "returnType": int, "parametersTypes": [list]},
        {"type": "function", "function": getMin, "returnType": int, "parametersTypes": [list]},
        {"type": "function", "function": sumElementsInt, "returnType": int, "parametersTypes": [float]},
        {"type": "function", "function": getMin, "returnType": int, "parametersTypes": [list]},
        {"type": "function", "function": getMin, "returnType": int, "parametersTypes": [list]},
    ]


def listFunctionsFloat():
    return [
        {"type": "function", "function": sumElementsFloat, "returnType": float, "parametersTypes": [float, float]},
        {"type": "function", "function": multiplyElementsFloat, "returnType": float, "parametersTypes": [float, float]},
        {"type": "function", "function": divideElementsFloat, "returnType": float, "parametersTypes": [float, float]}
    ]


################################################## NOTE return body ##################################################


def forLoop(body, start, stop, step):
    for index in range(start, stop, step):
        # evaluate(body, index)
        return
        

# def ifElse(body, conditions):
#     return
            
    


################################################## NOTE return array ##################################################


def swap1D(array, pos1, pos2):
    try:
        arrayCopy = array.copy()
        arrayCopy[pos1], arrayCopy[pos2] = arrayCopy[pos2], arrayCopy[pos1]
        return arrayCopy
    except:
        return arrayCopy


def sort1D(array, reverse):
    return  np.sort(array).tolist()[::-1] if reverse else np.sort(array).tolist()


def flip1D(array):
    return np.flip(array).tolist()


################################################## NOTE return int ##################################################


def length1D(array):
    return len(array)


def convertInt(number):
    return int(number)


def absolute(number):
    return abs(number)


def getMax(array):
    return max(array)


def getMin(array):
    return min(array)


def sumElementsInt(x, y):
    return int(x) + int(y)


def multiplyElementsInt(x, y):
    return int(x) * int(y)


def divideElementsInt(x, y):
    if y == 0:
        return 1
    return x // y


def modulo(x, y):
    return 1 if y == 0 else x % y


################################################## NOTE return float ##################################################


def sumElementsFloat(x, y):
    return x + y


def multiplyElementsFloat(x, y):
    return x * y


def divideElementsFloat(x, y):
    if y == 0:
        return 1
    return x / y