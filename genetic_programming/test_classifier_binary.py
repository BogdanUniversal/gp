#    This file is part of EAP.
# %%
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import random
import operator
import csv
import itertools

import numpy as np
import pandas as pd

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import umap

from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

import dalex as dx

# %%

# Read the spam list features and put it in a list of lists.
# The dataset is from http://archive.ics.uci.edu/ml/datasets/Spambase
# This example is a copy of the OpenBEAGLE example :
# http://beagle.gel.ulaval.ca/refmanual/beagle/html/d2/dbe/group__Spambase.html
# with open(r"C:\Users\bogda\Downloads\spambase\spambase.csv") as spambase:
#     spamReader = csv.reader(spambase)
#     spam = list(list(float(elem) for elem in row) for row in spamReader)

gender = pd.read_csv(r"C:\Users\bogda\Downloads\gender\gender_classification_v7.csv")
gender["gender"].replace({"Male": 1, "Female": 0}, inplace=True)

X, y = gender.iloc[:, :-1], gender.iloc[:, -1]

def foo(x, y):
    """
    STHIS IS SHIT
    """
    return x + y

# %%
# Group correlated columns
correlation_matrix = X.corr()
threshold = 0.5

correlated_groups = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            # Find or create a group for the correlated columns
            for group in correlated_groups:
                if (
                    correlation_matrix.columns[i] in group
                    or correlation_matrix.columns[j] in group
                ):
                    group.update(
                        [correlation_matrix.columns[i], correlation_matrix.columns[j]]
                    )
                    break
            else:
                correlated_groups.append(
                    set([correlation_matrix.columns[i], correlation_matrix.columns[j]])
                )

correlated_groups = [list(group) for group in correlated_groups]

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
umap_reducer = umap.UMAP(
    n_components=1, random_state=42, output_metric="euclidean"
)  # Enable inverse transform
# Fit the data
# for group in correlated_groups:
#     if len(group) > 1:  # Only apply UMAP if the group has more than one feature
#         umap_reducer = umap_reducer.fit(X[group])

#         # Add UMAP component to the dataset
#         colName = f"UMAP-{'-'.join(map(str, group))}"

#         X[colName] = umap_reducer.transform(X[group]).flatten()

#         X_train[colName] = umap_reducer.transform(X_train[group]).flatten()
#         X_train.drop(columns=group, inplace=True)  # Drop the original features

#         X_test[colName] = umap_reducer.transform(X_test[group]).flatten()
#         # X_test.drop(columns=group, inplace=True)  # Drop the original features
#     else:
#         # If the group has only one feature, skip UMAP and retain the feature
#         print(f"Skipping UMAP for single-feature group: {group}")


for group in correlated_groups:
    if len(group) > 1:
        # Scale number of components based on group size
        n_comp = 2  # Between 1 and 3 components
        
        umap_reducer = umap.UMAP(
            n_components=n_comp, 
            random_state=42, 
            output_metric="euclidean"
        )
        
        umap_reducer = umap_reducer.fit(X[group])
        
        # Handle multiple components if n_comp > 1
        umap_result = umap_reducer.transform(X[group])
        for i in range(n_comp):
            colName = f"UMAP-{'-'.join(map(str, group))}-{i}"
            X[colName] = umap_result[:, i]
            X_train[colName] = umap_reducer.transform(X_train[group])[:, i]
            X_test[colName] = umap_reducer.transform(X_test[group])[:, i]


# %%

scaler = StandardScaler()
scaler.set_output(transform="pandas")
scaler.fit(X[X_train.columns])  # Fit only on the training data


X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(
    X_test[X_train.columns]
)  # Use the same columns as in training

X_train_list = X_train_standardized.values.tolist()
X_test_list = X_test_standardized.values.tolist()
y_train_list = y_train.values.tolist()
y_test_list = y_test.values.tolist()

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped(
    "MAIN",
    itertools.repeat(
        float,
        gender.shape[1]
        - 1
        - sum([len(group) for group in correlated_groups])
        + 1 * len(correlated_groups),
    ),
    float,
    "IN",
)  # ATENTIE LA 1 * len(correlated_groups)

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)


# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)


# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", partial(random.uniform, 0, 100), float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

# %%

creator.create("FitnessMax", base.Fitness, weights=(-3.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSpambase(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Randomly sample 400 mails in the spam database

    # Evaluate the sum of correctly identified mail as spam
    # result = sum(bool(func(*mail[:57])) is bool(mail[57]) for mail in spam_samp)
    y_train_predict = [sigmoid(func(*row)) for row in X_train_list]

    return (mean_squared_error(y_train_list, y_train_predict), individual.height)


toolbox.register("evaluate", evalSpambase)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mut_shrink", gp.mutShrink)
toolbox.register("mut_eph", gp.mutEphemeral, mode="all")

random.seed(10)
pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# %%

import multiprocessing

if __name__ == "__main__":
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof)

# %%

# %%

func = toolbox.compile(hof[0])
results = []
for row in X_test_list:
    results.append(sigmoid(func(*row)))

# dfObserved = X_test.copy()
# dfPredicted = X_test.copy()

# dfObserved = scaler.inverse_transform(dfObserved, copy=True)
# dfPredicted = scaler.inverse_transform(dfPredicted, copy=True)

# dfObserved["Observed"] = y_test.values.tolist()
# dfPredicted["Predicted"] = results

# plotDFPredicted = pd.DataFrame(dfPredicted, columns=gender.columns)

# plotDFObserved = pd.DataFrame(dfObserved, columns=gender.columns)
# plotDFObserved.rename(columns={"gender": "Observed"}, inplace=True)

# plotDFObserved["Predicted"] = plotDFPredicted["gender"]

plotDFObserved = pd.DataFrame(
    {"Observed": y_test.values.tolist(), "Predicted": results}
)
plotDFObserved["Predicted"] = plotDFObserved["Predicted"].apply(
    lambda x: 1 if x > 0.5 else 0
)

plotDFObserved.sort_values(by="Observed", ascending=True, inplace=True)

plotDFObserved.reset_index(drop=True, inplace=True)
# %%

import plotly.express as px

plt = px.scatter(plotDFObserved[["Observed", "Predicted"]])
plt.show()


# %%
# import warnings
# warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")


def predict_function(model, data):
    cols = []
    for group in correlated_groups:
        if len(group) > 1:  # Only apply UMAP if the group has more than one feature
            colName = f"UMAP-{'-'.join(map(str, group))}"
            cols.append(colName)

            data[colName] = umap_reducer.transform(data[group]).flatten()
        else:
            # If the group has only one feature, skip UMAP and retain the feature
            print(f"Skipping UMAP for single-feature group: {group}")

    data_standardized = scaler.transform(data[X_train.columns])  # Standardize the data

    data.drop(columns=cols, inplace=True)  # Drop the new features

    dataList = data_standardized.values.tolist()
    func = toolbox.compile(model[0])  # Compile the best individual
    return np.array([sigmoid(func(*row)) for row in dataList])


X_test_predict = X_test.iloc[:, : -len(correlated_groups)].copy()

# Initialize the explainer
explainer = dx.Explainer(
    model=hof,  # Pass the Hall of Fame
    data=X_test_predict,  # Test data
    y=y_test,  # True labels
    predict_function=predict_function,  # Custom predict function
    model_type="classification",
    label="Genetic Programming Model",
)
performance = explainer.model_performance()
performance.plot()

single_explanation = explainer.predict_parts(X_test_predict.iloc[1], type="shap")
single_explanation.plot()

profile = explainer.model_profile()
profile.plot()

# %%
