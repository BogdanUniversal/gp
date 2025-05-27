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