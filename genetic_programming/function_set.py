# from helper import evaluate
import numpy as np
import uuid
from sympy import isprime


def listFunctions1D():
    return [
        {
            "type": "function",
            "function": swap1D,
            "returnType": list,
            "parametersTypes": [list, int, int],
        },
        {
            "type": "function",
            "function": sort1D,
            "returnType": list,
            "parametersTypes": [list, bool],
        },
        {
            "type": "function",
            "function": flip1D,
            "returnType": list,
            "parametersTypes": [list],
        },
    ]


def listFunctionsInt():
    return [
        {
            "type": "function",
            "function": length1D,
            "returnType": int,
            "parametersTypes": [list],
        },
        {
            "type": "function",
            "function": convertInt,
            "returnType": int,
            "parametersTypes": [int],
        },
        {
            "type": "function",
            "function": absolute,
            "returnType": int,
            "parametersTypes": [int],
        },
        {
            "type": "function",
            "function": getMax,
            "returnType": int,
            "parametersTypes": [list],
        },
        {
            "type": "function",
            "function": getMin,
            "returnType": int,
            "parametersTypes": [list],
        },
        {
            "type": "function",
            "function": sumElementsInt,
            "returnType": int,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": getMin,
            "returnType": int,
            "parametersTypes": [list],
        },
        {
            "type": "function",
            "function": getMin,
            "returnType": int,
            "parametersTypes": [list],
        },
    ]


def listFunctionsFloat():
    return [
        {
            "type": "function",
            "function": sumElementsFloat,
            "returnType": float,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": multiplyElementsFloat,
            "returnType": float,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": divideElementsFloat,
            "returnType": float,
            "parametersTypes": [float, float],
        },
    ]


def listFunctionsBoolean():
    return [
        {
            "type": "function",
            "function": numericComparisonLarger,
            "returnType": bool,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": numericComparisonLargerEqual,
            "returnType": bool,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": numericComparisonEqual,
            "returnType": bool,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": numericComparisonNotEqual,
            "returnType": bool,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": numericComparisonDivisible,
            "returnType": bool,
            "parametersTypes": [float, float],
        },
        {
            "type": "function",
            "function": logicalComparisonAnd,
            "returnType": bool,
            "parametersTypes": [bool, bool],
        },
        {
            "type": "function",
            "function": logicalComparisonOr,
            "returnType": bool,
            "parametersTypes": [bool, bool],
        },
        {
            "type": "function",
            "function": logicalComparisonXor,
            "returnType": bool,
            "parametersTypes": [bool, bool],
        },
        {
            "type": "function",
            "function": numericPredicatesEven,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesOdd,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesPositive,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesNegative,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesZero,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesNotZero,
            "returnType": bool,
            "parametersTypes": [float],
        },
        {
            "type": "function",
            "function": numericPredicatesPrime,
            "returnType": bool,
            "parametersTypes": [float],
        },
    ]


################################################## NOTE return array ##################################################


def swap1D(array, pos1, pos2):
    arrayCopy = array.copy()
    arrayCopy[pos1], arrayCopy[pos2] = arrayCopy[pos2], arrayCopy[pos1]
    return arrayCopy


def sort1D(array, reverse):
    return np.sort(array).tolist()[::-1] if reverse else np.sort(array).tolist()


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


################################################## NOTE return bool ##################################################


def numericComparisonLarger(x, y):
    return x > y

def numericComparisonLargerEqual(x, y):
    return x >= y

def numericComparisonEqual(x, y):
    return x == y

def numericComparisonNotEqual(x, y):
    return x != y

def numericComparisonDivisible(x, y):
    return x % y == 0 if y != 0 else 0


def logicalComparisonAnd(x, y):
    return x and y

def logicalComparisonOr(x, y):
    return x or y

def logicalComparisonXor(x, y):
    return bool(x) != bool(y)


def numericPredicatesEven(x):
    return x % 2 == 0

def numericPredicatesOdd(x):
    return x % 2 != 0

def numericPredicatesPositive(x):
    return x > 0

def numericPredicatesNegative(x):
    return x < 0

def numericPredicatesZero(x):
    return x == 0

def numericPredicatesNotZero(x):
    return x != 0

def numericPredicatesPrime(x):
    return isprime(x)

