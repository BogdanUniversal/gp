# %%
import random
import operator
import itertools

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler

# %%

# Example data


df = pd.read_pickle("./testlist.pkl")
X = df["X"]  # Features (all columns except the last one)
y = df["y"]  # Target (last column)

X_df = pd.DataFrame(X.tolist(), columns=[f"X{i}" for i in range(len(X.iloc[0]))])
y_df = pd.DataFrame(y.tolist(), columns=[f"Y{i}" for i in range(len(y.iloc[0]))])

df_combined = pd.concat([X_df, y_df], axis=1)

# scaler = StandardScaler()
# scaler.set_output(transform="pandas")
# scaler.fit(df_combined)

# df_combined_scaled = scaler.transform(df_combined)
# X_scaled = [list(row) for row in df_combined_scaled[X_df.columns].values.tolist()]
# y_scaled = [list(row) for row in df_combined_scaled[y_df.columns].values.tolist()]

y = [list(row) for row in df_combined[y_df.columns].values.tolist()]
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()


# %%

# Define the primitive set for integers
pset = gp.PrimitiveSetTyped(
    "MAIN",
    itertools.repeat(int, len(X.iloc[0])),  # Input: list of integers
    list,  # Output: list of integers
    "IN",
)

# pset = gp.PrimitiveSetTyped(
#     "MAIN",
#     [list],  # Input: a single list of integers
#     list,    # Output: a list of integers
#     "IN"
# )

# Add integer-specific primitives
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)


def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)


# Define a protected division function for integers
def protectedDivInt(left, right):
    try:
        return left // right  # Integer division
    except ZeroDivisionError:
        return 1


pset.addPrimitive(protectedDivInt, [float, float], float)


# Add a primitive to create a list from integers
def create_list(a, b, c, d, e, f, g, h, i, j):
    return [a, b, c, d, e, f, g, h, i, j]


pset.addPrimitive(
    create_list,
    [float, float, float, float, float, float, float, float, float, float],
    list,
)


def getElement(lst, index):
    """Get an element from a list by index, with bounds checking."""
    # print(f"Getting element at index {index} from list {lst}")
    if 0 <= index < len(lst):
        return lst[int(index)]
    else:
        return 0  # Return 0 if index is out of bounds


pset.addPrimitive(getElement, [list, int], float)

# # Update primitives to work with lists
def add_lists(a, b):
    return [x + y for x, y in zip(a, b)]

def sub_lists(a, b):
    return [x - y for x, y in zip(a, b)]

def mul_lists(a, b):
    return [x * y for x, y in zip(a, b)]

pset.addPrimitive(add_lists, [list, list], list)
pset.addPrimitive(sub_lists, [list, list], list)
pset.addPrimitive(mul_lists, [list, list], list)


class FloatFunction:
    """A placeholder type for a function that takes a float and returns a float."""
    pass

class Double(FloatFunction):
    def __call__(self, x):
        return x * 2
    
    def __name__(self):
        return "Double"

class Negate(FloatFunction):
    def __call__(self, x):
        return -x
    
    def __name__(self):
        return "Negate"

# pset.addPrimitive(Double(), [float], FloatFunction)
# pset.addPrimitive(Negate(), [float], FloatFunction)
# pset.addTerminal(Double(), FloatFunction)
# pset.addTerminal(Negate(), FloatFunction)
    
    # Add callable terminals of type FloatFunction

def for_each(lst, i, j):
    """Apply a function to each element of a list."""
    if i < 0 or j < 0 or i >= len(lst) or j >= len(lst):
        return lst  # Return the list unchanged if indices are out of bounds
    for _ in enumerate(lst):
        lst[int(i)], lst[int(j)] = lst[int(j)], lst[int(i)]  # Swap elements in the list
    return lst


def for_for_each(lst, i, j):
    """Apply a function to each element of a list."""
    if i < 0 or j < 0 or i >= len(lst) or j >= len(lst):
        return lst  # Return the list unchanged if indices are out of bounds
    for _ in enumerate(lst):
        for _ in enumerate(lst):
            lst[int(i)], lst[int(j)] = lst[int(j)], lst[int(i)]  # Swap elements in the list
    return lst


# class ForFunction:
#     """A placeholder type for a function that takes a list and returns a list."""
#     pass

# class Swap(ForFunction):
#     def __call__(self, lst, i, j):
#         if i < 0 or j < 0 or i >= len(lst) or j >= len(lst):
#             return lst  # Return the list unchanged if indices are out of bounds
#         lst[int(i)], lst[int(j)] = lst[int(j)], lst[int(i)]  # Swap elements in the list
#         return lst
    
#     def __name__(self):
#         return "Swap"

# class Negate(ForFunction):
#     def __call__(self):
#         return [-1 * x for x in lst]  # Negate each element in the list
    
#     def __name__(self):
#         return "Negate"


# def for_for_each_general(lst, func):
#     """Apply a function to each element of a list."""
#     for i, el_i in enumerate(lst):
#         for j, el_j in enumerate(lst):
#             func(lst, i, j)  # Apply the function to each pair of elements
#     return lst



pset.addPrimitive(for_each, [list, int, int], list)
pset.addPrimitive(for_for_each, [list, int, int], list)

# pset.addPrimitive(Swap(), [list, int, int], ForFunction)
# pset.addTerminal(Swap()([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 1), ForFunction)
# pset.addPrimitive(for_for_each_general, [list, ForFunction], list)


# Add terminals
pset.addEphemeralConstant(
    "rand100", lambda: random.randint(0, 10), int
)  # Random integers
pset.addEphemeralConstant(
    "randfloat", lambda: random.uniform(-10, 10), float
)  # Random integers
pset.addEphemeralConstant(
    "rand10", lambda: np.random.randn(10).tolist(), list
)  # Random list floats
pset.addEphemeralConstant(
    "randintlist", lambda: np.random.randint(1, 100, 10).tolist(), list
)  # Random list integers
pset.addTerminal(0, int)
pset.addTerminal(1, int)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)


# %%
# Define the evaluation function
def evalFunction(individual):
    # Transform the tree expression into a callable function
    func = toolbox.compile(expr=individual)

    # Debugging the compiled function
    y_train_predict = []
    for row in X_train:
        result = func(*row)  # Pass the entire row as a single argument
        y_train_predict.append(result)

        # try:
        # result = func(row)  # Pass the entire row as a single argument
        # y_train_predict.append(result)
        # except Exception as e:
        #     print(f"Error with input {row}: {e}")
        #     y_train_predict.append([0] * len(row))  # Default to a list of zeros

    # Calculate mean squared error between predicted and actual outputs
    mse = mean_squared_error(y_train, y_train_predict)
    return (mse,)  # Return as a tuple


# %%
# Register the toolbox

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalFunction)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# %%
# Run the genetic programming algorithm
random.seed(42)
pop = toolbox.population(n=100)
hof = tools.HallOfFame(5)

def get_fitness_values(ind):
    return ind.fitness.values

stats = tools.Statistics(get_fitness_values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

import multiprocessing

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100000, stats, halloffame=hof)

# Compile the best individual
best_func = toolbox.compile(expr=hof[0])

# Test the best individual
results = [best_func(*row) for row in X_test]
print("Predicted:", results)
print("Actual:", y_test)
# %%
