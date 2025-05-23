from functools import partial
import numpy as np


TERMINALS = [
    {"id": "true", "name": "True", "type": "Constant", "function": True},
    {
        "id": "false",
        "name": "False",
        "type": "Constant",
        "function": False,
        "out": bool,
    },
    {
        "id": "one",
        "name": "One",
        "type": "Constant",
        "function": 1,
        "out": float,
    },
    {
        "id": "minus_one",
        "name": "Minus One",
        "type": "Constant",
        "function": -1,
        "out": float,
    },
    {
        "id": "rand_unif_100",
        "name": "Random Uniform (0 - 100)",
        "type": "Terminal",
        "function": partial(np.random.uniform, 0, 100),
        "out": float,
    },
    {
        "id": "rand_unif_minus",
        "name": "Random Uniform (-1 - 1)",
        "type": "Terminal",
        "function": partial(np.random.uniform, -1, 1),
        "out": float,
    },
    {
        "id": "rand_gauss_0",
        "name": "Random Normal (0 Mean)",
        "type": "Terminal",
        "function": np.random.normal,
        "out": float,
    },
    {
        "id": "rand_wald",
        "name": "Random Wald (1 Mean)",
        "type": "Terminal",
        "function": partial(np.random.wald, 1, 1),
        "out": float,
    },
    {
        "id": "rand_pareto",
        "name": "Random Pareto (1 Shape)",
        "type": "Terminal",
        "function": partial(np.random.pareto, 1),
        "out": float,
    },
    {
        "id": "rand_poission",
        "name": "Random Poisson (2 Lam)",
        "type": "Terminal",
        "function": partial(np.random.poisson, 2),
        "out": float,
    },
]
