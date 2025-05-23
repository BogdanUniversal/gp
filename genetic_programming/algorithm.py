from functools import partial
from deap import base, creator, tools
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import umap
from general_set import LOSSES_SET, MUTATION_SET, SELECTION_SET
from deap import gp, algorithms, base, creator, tools
import itertools
from genetic_programming.primitive_set_gp import PRIMITIVES
from genetic_programming.terminal_set_gp import TERMINALS
from mvc.model.dataset_cache import dataset_cache
from mvc.model.parameters_cache import parameters_cache


creator.create("FitnessMin", base.Fitness, weights=(-4.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def getCorrelatedGroups(correlation_matrix, threshold=0.6):
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
                            [
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                            ]
                        )
                        break
                else:
                    correlated_groups.append(
                        set(
                            [
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                            ]
                        )
                    )

    return [list(group) for group in correlated_groups]


def custom_mutation(rng, mutations, pset, expr, ms, treeDepth):
    selected_mutation = rng.choice(mutations)

    if selected_mutation["id"] == "mutUniform":
        return partial(selected_mutation["function"], expr=expr, pset=pset)
    elif selected_mutation["id"] == "mutEphemeral":
        return partial(selected_mutation["function"], mode="all")
    elif selected_mutation["id"] == "mutSemantic":
        return partial(
            selected_mutation["function"],
            gen_func=gp.genHalfAndHalf,
            pset=pset,
            min_=0,
            max_=max(4, treeDepth // 3),
        )


def run_genetic_algorithm_pipeline(user_id):
    dataset = dataset_cache.get(str(user_id)).copy()
    parameters = parameters_cache.get(user_id)

    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed.entropy)

    classificationOk = (
        True
        if parameters["lossFunction"]["id"] in LOSSES_SET[0] + LOSSES_SET[1]
        else False
    )

    label_encoder = LabelEncoder()
    if classificationOk:
        dataset[parameters["selectedLabel"]] = label_encoder.fit_transform(
            dataset[parameters["selectedLabel"]]
        )

    X = dataset.drop(columns=[parameters["selectedLabel"]])
    y = dataset[parameters["selectedLabel"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed.entropy
    )

    scalerX = StandardScaler()
    scalerX.set_output(transform="pandas")

    scalerY = StandardScaler()
    scalerY.set_output(transform="pandas")

    scalerX.fit(X)
    X_train_standardized = scalerX.transform(X_train)
    X_test_standardized = scalerX.transform(X_test)

    X_train_list = X_train_standardized.values.tolist()
    X_test_list = X_test_standardized.values.tolist()

    if not classificationOk:
        scalerY.fit(y.values.reshape(-1, 1))
        y_train_standardized = scalerY.transform(y_train.values.reshape(-1, 1))
        y_test_standardized = scalerY.transform(y_test.values.reshape(-1, 1))
        y_train_list = y_train_standardized.values.tolist()
        y_test_list = y_test_standardized.values.tolist()
    else:
        y_train_list = y_train.values.tolist()
        y_test_list = y_test.values.tolist()

    correlation_matrix = X.corr(method=parameters["corrOpt"].lower())
    correlated_groups = getCorrelatedGroups(correlation_matrix, 0.6)

    def getGroupNComponent(group):
        return 1 if len(group) == 2 else 2 if len(group) < 5 else 3

    for group in correlated_groups:
        if len(group) > 1:
            n_comp = getGroupNComponent(group)

            reducer = (
                umap.UMAP(
                    n_components=n_comp,
                    random_state=seed.entropy,
                    output_metric="euclidean",  # CHECK IF CORRECT NOTE
                )
                if parameters["dimRedOpt"] == "UMAP"
                else PCA(n_components=n_comp, random_state=seed.entropy)
            )

            reducer = reducer.fit(X[group])
            result = reducer.transform(X[group])
            for i in range(n_comp):
                colName = f"REDUCED-{'-'.join(map(str, group))}-{i}"
                X[colName] = result[:, i]
                X_train[colName] = reducer.transform(X_train[group])[:, i]
                X_test[colName] = reducer.transform(X_test[group])[:, i]

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        itertools.repeat(
            float,
            dataset.shape[1]
            - 1
            - sum([len(group) for group in correlated_groups])
            + [getGroupNComponent(group) for group in correlated_groups],
        ),
        float,
        "IN",
    )

    for funParam in parameters["functions"]:
        if funParam["type"] == "Primitive":
            fun = [f for f in PRIMITIVES if f["id"] == funParam["id"]][0]
            pset.addPrimitive(
                fun["function"],
                fun["in"],
                fun["out"],
                name=fun["name"],
            )
        elif funParam["type"] == "Terminal":
            fun = [f for f in TERMINALS if f["id"] == funParam["id"]][0]
            pset.addEphemeralConstant(fun["name"], fun["function"], fun["out"])
        elif funParam["type"] == "Constant":
            fun = [f for f in TERMINALS if f["id"] == funParam["id"]][0]
            pset.addTerminal(fun["function"], fun["out"])

    toolbox = base.Toolbox()
    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=parameters["treeDepth"]
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalSpambase)
    toolbox.register(
        "select",
        [s for s in SELECTION_SET if s["id"] == parameters["selectionMethod"]["id"]][0][
            "function"
        ],
        tournsize=5,
    )
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register(
        "expr_mut", gp.genHalfAndHalf, min_=0, max_=max(4, parameters["treeDepth"] // 3)
    )

    for mutParam in parameters["mutationFunction"]:
        mut = [m for m in MUTATION_SET if m["id"] == mutParam["id"]][0]
        toolbox.register(
            "mutate",
            mut["function"],
            expr=toolbox.expr_mut,
            pset=pset,
            **mut.get("kwargs", {}),
        )
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mut_shrink", gp.mutShrink)
    toolbox.register("mut_eph", gp.mutEphemeral, mode="all")

    # Define the callback function to send updates
    def update_callback(gen, stats, best_individual):
        # Extract data to send to frontend
        update_data = {
            "generation": gen,
            "best_fitness": float(stats["min"]),  # Convert numpy types to native Python
            "avg_fitness": float(stats["avg"]),
            "std_dev": float(stats["std"]),
        }

        # If generation is multiple of 5 or it's the last generation
        # include the best individual (to reduce data traffic)
        if gen % 5 == 0 or gen == parameters["ngen"]:
            update_data["best_individual"] = str(best_individual)

        # Send data via WebSocket
        socketio.emit("training_update", update_data)

    # Run the algorithm with callback
    eaSimpleWithCallback(
        pop,
        toolbox,
        parameters["cxpb"],
        parameters["mutpb"],
        parameters["ngen"],
        stats,
        halloffame=hof,
        callback=update_callback,
    )

    # Return the final results
    return hof[0]


def eaSimpleWithCallback(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callback=None,
    verbose=True,
):
    """This algorithm reproduces the simplest evolutionary algorithm with real-time callback updates.

    :param callback: Function called after each generation with (gen, statistics, best_individual)
    """
    from deap import tools, algorithms

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Call the callback with initial generation data
    if callback:
        best_ind = tools.selBest(population, 1)[0] if population else None
        callback(0, record, best_ind)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Call the callback with updated data
        if callback:
            best_ind = tools.selBest(population, 1)[0] if population else None
            callback(gen, record, best_ind)

    return population, logbook
