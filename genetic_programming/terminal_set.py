import numpy as np
import math


def terminal1Dto1D():
    return ["ephemeralRandomConstant"]


def listTerminalInt(maxListLength, rng: np.random.default_rng):
    return [
        {"type": "terminal", "terminal": "zero", "returnType": int, "value": 0},
        {"type": "terminal", "terminal": "one", "returnType": int, "value": 1},
        {"type": "terminal", "terminal": "two", "returnType": int, "value": 2},
        {"type": "terminal", "terminal": "three", "returnType": int, "value": 3},
        {"type": "terminal", "terminal": "five", "returnType": int, "value": 5},
        {"type": "terminal", "terminal": "seven", "returnType": int, "value": 7},
        {"type": "terminal", "terminal": "ten", "returnType": int, "value": 10},
        {"type": "terminal", "terminal": "hex", "returnType": int, "value": 16},
        {
            "type": "terminal",
            "terminal": "randomInt",
            "returnType": int,
            "value": generateRandomInt(maxListLength, rng),
        },
    ]


def listTerminalFloat(maxListLength, rng: np.random.default_rng):
    return [
        {
            "type": "terminal",
            "terminal": "randomUniform",
            "returnType": float,
            "value": generateRandomUniform(maxListLength, rng),
        },
        {
            "type": "terminal",
            "terminal": "randomGaussian",
            "returnType": float,
            "value": generateRandomGaussian(maxListLength, rng),
        },
        {"type": "terminal", "terminal": "pi", "returnType": float, "value": math.pi},
        {"type": "terminal", "terminal": "euler", "returnType": float, "value": math.e},
    ]


def listTerminalBoolean(rng: np.random.default_rng):
    return [
        {
            "type": "terminal",
            "terminal": "randomBoolean",
            "returnType": bool,
            "value": generateRandomBoolean(rng),
        },
    ]


def generateRandomInt(maxListLength, rng: np.random.default_rng):
    choice = rng.choice([1, 2], p=[0.9, 0.1])
    if choice == 1:
        return int(rng.integers(0, maxListLength))
    return int(rng.integers((-1) * maxListLength, maxListLength))


def generateRandomUniform(maxListLength, rng: np.random.default_rng):
    choice = rng.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
    if choice == 1:
        return rng.uniform(0, maxListLength)
    elif choice == 2:
        return rng.uniform(0, 1)
    return rng.uniform((-1) * maxListLength, maxListLength)


def generateRandomGaussian(maxListLength, rng: np.random.default_rng):
    choice = rng.choice([1, 2], p=[0.7, 0.3])
    if choice == 1:
        return rng.normal(1)
    return rng.normal(maxListLength / 2)


def generateRandomBoolean(rng: np.random.default_rng):
    return rng.choice([True, False])
