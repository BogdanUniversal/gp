from functools import partial
import operator
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    hinge_loss,
    mean_squared_error,
    mean_absolute_error,
)
from genetic_programming.primitive_set_gp import *
from genetic_programming.terminal_set_gp import *

from mvc.model.dataset_cache import dataset_cache

LOSSES = [
    [
        {"id": "bce", "name": "Binary Cross Entropy", "function": log_loss},
        {"id": "hinge", "name": "Hinge loss", "function": hinge_loss},
    ],
    [{"id": "cce", "name": "Categorical Cross Entropy", "function": log_loss}],
    [
        {"id": "mse", "name": "Mean Squared Error", "function": mean_squared_error},
        {"id": "mae", "name": "Mean Absolute Error", "function": mean_absolute_error},
    ],
]


def getLossFunctions(userId, labelColumn):
    dataset = dataset_cache.get(str(userId))
    labelData = dataset[labelColumn]

    isNumeric = pd.api.types.is_numeric_dtype(labelData)

    uniqueValues = labelData.nunique()

    if isNumeric:
        if uniqueValues > 20:
            return LOSSES[2]

    if uniqueValues == 2:
        return LOSSES[0]
    elif uniqueValues > 2:
        return LOSSES[1]
    else:
        return LOSSES[2]
