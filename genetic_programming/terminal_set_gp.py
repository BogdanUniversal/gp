
from functools import partial
import numpy as np


TERMINALS = [
    {
        "id": "true",
        "name": "True",
        "type": "Boolean",
        "function": True
    },
    {
        "id": "false",
        "name": "False",
        "type": "Boolean",
        "function": False,
    },
    {
        "id": "rand_unif_100",
        "name": "Random Uniform (0 - 100)",
        "type": "Float",
        "function": partial(np.random.uniform, 0, 100),
    },
    {
        "id": "rand_unif_minus",
        "name": "Random Uniform (-1 - 1)",
        "type": "Float",
        "function": partial(np.random.uniform, -1, 1),
    },
    {
        "id": "rand_gauss_0",
        "name": "Random Normal (0 Mean)",
        "type": "Float",
        "function": np.random.normal,
    },
    {
        "id": "rand_wald",
        "name": "Random Wald (1 Mean)",
        "type": "Float",
        "function": partial(np.random.wald, 1, 1),
    },
    {
        "id": "rand_pareto",
        "name": "Random Pareto (1 Shape)",
        "type": "Float",
        "function": partial(np.random.pareto, 1),
    },
    {
        "id": "rand_poission",
        "name": "Random Poisson (2 Lam)",
        "type": "Float",
        "function": partial(np.random.poisson, 2),
    },
]