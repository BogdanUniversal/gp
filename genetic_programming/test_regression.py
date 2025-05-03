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

import math
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
import time

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the spam list features and put it in a list of lists.
# The dataset is from http://archive.ics.uci.edu/ml/datasets/Spambase
# This example is a copy of the OpenBEAGLE example :
# http://beagle.gel.ulaval.ca/refmanual/beagle/html/d2/dbe/group__Spambase.html
# with open(r"C:\Users\bogda\Downloads\spambase\spambase.csv") as spambase:
#     spamReader = csv.reader(spambase)
#     spam = list(list(float(elem) for elem in row) for row in spamReader)

studentPerformance = pd.read_csv(r"C:\Users\bogda\Downloads\Student_Performance.csv")
studentPerformance["Extracurricular Activities"].replace({"Yes": 1, "No": 0}, inplace=True)

scaler = StandardScaler()
scaler.fit(studentPerformance)
scaler.set_output(transform = "pandas")

studentPerformanceStandardized = scaler.transform(studentPerformance)

X = studentPerformanceStandardized.iloc[:, :-1]
y = studentPerformanceStandardized.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_list = X_train.values.tolist()
X_test_list = X_test.values.tolist()
y_train_list = y_train.values.tolist()
y_test_list = y_test.values.tolist()

# %%

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 5), float, "IN")

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
pset.addEphemeralConstant("rand100", partial(random.uniform, -1, 1), float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

# %%

creator.create("FitnessMax", base.Fitness, weights=(1.0, -2.0, -1.0))
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
    y_train_predict = [func(*row) for row in X_train_list]
    # y_test_predict = [func(*row) for row in X_test_list]
    
    
    # if np.any(np.isnan(y_train_predict)):
    #     print("y_train_predict contains NaN!")
    # if np.any(np.isinf(y_train_predict)):
    #     print("y_train_predict contains infinity!")
    
    try:
        r2Train = r2_score(y_train_list, y_train_predict)
        mseTrain = mean_squared_error(y_train_list, y_train_predict)
        # mseTest = r2_score(y_test_list, y_test_predict)
        height = individual.height
        # print(f"Height: {height}, MSE Train: {mseTrain}, MSE Test: {mseTest}")
    except ValueError:
        r2Train = math.inf
        mseTrain = math.inf
        height = math.inf
    
    # print(f"MSE Train: {mseTrain}, MSE Test: {mseTest}")

    return (r2Train, mseTrain, height)


toolbox.register("evaluate", evalSpambase)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# %%
def ea():
    random.seed(10)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof, verbose=True)

# %%

from scoop import futures
# import multiprocessing

if __name__ == "__main__":
    toolbox.register("map", futures.map)
    ea()
    

# %% 

func = toolbox.compile(hof[0])
results=[]
for row in X_test_list:
    results.append(func(*row))

dfObserved = X_test.copy()
dfObserved["Observed"] = y_test.values.tolist()

dfPredicted = X_test.copy()
dfPredicted["Predicted"] = results

dfObserved = scaler.inverse_transform(dfObserved, copy=True)
dfPredicted = scaler.inverse_transform(dfPredicted, copy=True)

plotDFPredicted = pd.DataFrame(dfPredicted, columns=studentPerformance.columns)

plotDFObserved = pd.DataFrame(dfObserved, columns=studentPerformance.columns)
plotDFObserved.rename(columns={"Performance Index": "Observed"}, inplace=True)

plotDFObserved["Predicted"] = plotDFPredicted["Performance Index"]



plotDFObserved.sort_values(by="Observed", ascending=True, inplace=True)

plotDFObserved.reset_index(drop=True, inplace=True)
# %%

import plotly.express as px
plt = px.scatter(plotDFObserved[["Observed", "Predicted"]])
plt.show()




