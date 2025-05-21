from functools import partial
import operator
import numpy as np
import pandas as pd
from genetic_programming.primitive_set_gp import *
from genetic_programming.terminal_set_gp import *
from genetic_programming.general_set import *


from mvc.model.dataset_cache import dataset_cache


def getLossFunctions(userId, labelColumn):
    dataset = dataset_cache.get(str(userId))
    labelData = dataset[labelColumn]

    isNumeric = pd.api.types.is_numeric_dtype(labelData)

    uniqueValues = labelData.nunique()

    if isNumeric:
        if uniqueValues > 20:
            return LOSSES_SET[2]

    if uniqueValues == 2:
        return LOSSES_SET[0]
    elif uniqueValues > 2:
        return LOSSES_SET[1]
    else:
        return LOSSES_SET[2]


def getTerminalsPrimitives():
    return PRIMITIVES + TERMINALS