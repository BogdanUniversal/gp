from functools import partial
import operator
import numpy as np
import pandas as pd
from genetic_programming.primitive_set_gp import *
from genetic_programming.terminal_set_gp import *
from genetic_programming.general_set import *
from mvc.model.dataset_cache import dataset_cache


def getTerminalsPrimitives():
    return PRIMITIVES + TERMINALS


def createModel(model):
    """
    Create a view for the model.
    """
    return {
        "id": str(model.id),
        "user_id": str(model.user_id),
        "model_file_name": model.model_file_name,
        "dim_red_file_name": model.dim_red_file_name,
        "plots_path": model.plots_path,
        "upload_date": model.upload_date.isoformat(),
    }