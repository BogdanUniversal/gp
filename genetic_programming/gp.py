# %% Polymorphic Higher-Order Type System

import math
import numpy as np
import pandas as pd
from function_set import *
from terminal_set import *
from tree import Node, Tree
from helper import *

# %%

input = [10,4,9,2,1,6,3,7,5,8]
output = [1,2,3,4,5,6,7,8,9,10]

SEED = np.random.SeedSequence()

rng = np.random.default_rng(SEED)

environment = {int: [], float: [], bool: [], list: []}

a = generateBody(environment, 10, rng, 50)

# asb = generateTree(rng, 0, 10, int)

