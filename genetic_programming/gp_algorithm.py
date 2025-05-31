from functools import partial
import dill
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
import umap
from genetic_programming.general_set import MUTATION_SET, SELECTION_SET
from deap import gp, algorithms, creator, base, tools
import dalex as dx
import itertools
from genetic_programming.primitive_set_gp import PRIMITIVES
from genetic_programming.terminal_set_gp import TERMINALS
from mvc.model.socket import socket_cache
from mvc.model.model import Model
# from mvc.model.dataset_cache import dataset_cache
import os


def getCorrelatedGroups(correlation_matrix, threshold=0.6):
    correlated_groups = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
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


def custom_mutation(individual, rng, mutations, pset, expr):
    selected_mutation = rng.choice(mutations)

    if selected_mutation["id"] == "mutUniform":
        return selected_mutation["function"](individual, expr=expr, pset=pset)
    elif selected_mutation["id"] == "mutEphemeral":
        return selected_mutation["function"](individual, mode="all")
    elif selected_mutation["id"] == "mutNodeReplacement":
        return selected_mutation["function"](
            individual,
            pset=pset,
        )
    elif selected_mutation["id"] == "mutInsert":
        return selected_mutation["function"](
            individual,
            pset=pset,
        )
    elif selected_mutation["id"] == "mutShrink":
        return selected_mutation["function"](
            individual,
        )
    else:
        raise ValueError(f"Unknown mutation type: {selected_mutation['id']}")


def identify_categorical_columns(dataset, selected_label):
    categorical_cols = []
    for col in dataset.columns:
        if col == selected_label:
            continue

        if dataset[col].nunique() <= 20 and dataset[col].nunique() > 2:
            categorical_cols.append(col)
            continue

        dtype = dataset[col].dtype
        if (
            dtype == "object"
            or dtype.name == "category"
            or dtype == "bool"
            or (np.issubdtype(dtype, np.integer) and dataset[col].nunique() <= 30)
        ):
            categorical_cols.append(col)
    return categorical_cols


def update_callback(userId, gen, stats):
    update_data = {
        "generation": gen,
        "best_fitness": round(float(stats["min"]), 4),
        "avg_fitness": round(float(stats["avg"]), 4),
        "max_fitness": round(float(stats["max"]), 4),
        "std_dev": round(float(stats["std"]), 4),
    }

    socket_cache.emit(
        userId,
        "training_update",
        update_data,
    )


def eaSimpleWithCallback(
    userId,
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callback=True,
    verbose=True,
):
    """This algorithm reproduces the simplest evolutionary algorithm with real-time callback updates.

    :param callback: Function called after each generation with (gen, statistics, best_individual)
    """
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
        # best_ind = tools.selBest(population, 1)[0] if population else None
        update_callback(userId, 0, record)

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
            # best_ind = tools.selBest(population, 1)[0] if population else None
            update_callback(userId, gen, record)

    return population, logbook


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def algorithm(model: Model, datasetCached: pd.DataFrame):
    # dataset = dataset_cache.get(str(model.user_id)).copy()
    dataset = datasetCached.copy()
    parameters = model.parameters

    dataset = dataset[parameters["selectedFeatures"] + [parameters["selectedLabel"]]]
    dataset.dropna(inplace=True)

    seed = np.random.SeedSequence()
    seed_restricted = int(seed.entropy) % (2**32 - 1)
    rng = np.random.default_rng(seed.entropy)

    classificationOk = True if parameters["objective"] == "Classification" else False

    label_encoder = LabelEncoder()
    if classificationOk:
        dataset[parameters["selectedLabel"]] = label_encoder.fit_transform(
            dataset[parameters["selectedLabel"]]
        )

    target_encode_categorial = TargetEncoder(
        target_type="binary" if classificationOk else "continuous"
    )
    columnsToEncode = identify_categorical_columns(dataset, parameters["selectedLabel"])
    if len(columnsToEncode):
        target_encode_categorial.fit(
            dataset[columnsToEncode],
            dataset[parameters["selectedLabel"]],
        )

    X = dataset[parameters["selectedFeatures"]].copy()
    y = dataset[parameters["selectedLabel"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_restricted
    )

    X_encoded = X.copy()
    X_encoded[columnsToEncode] = target_encode_categorial.transform(
        X_encoded[columnsToEncode]
    )
    correlation_matrix = X_encoded.corr(method=parameters["corrOpt"].lower())
    correlated_groups = getCorrelatedGroups(correlation_matrix, 0.5)

    scalerX = StandardScaler()
    scalerX.set_output(transform="pandas")
    scalerY = StandardScaler()
    scalerY.set_output(transform="pandas")

    scalerX.fit(X_encoded)
    X_standardized = scalerX.transform(X_encoded)
    scalerY.fit(y.values.reshape(-1, 1))
    y_standardized = scalerY.transform(y.values.reshape(-1, 1))

    if len(columnsToEncode):
        X_train_standardized = X_train.copy()
        X_test_standardized = X_test.copy()
        X_train_standardized[columnsToEncode] = target_encode_categorial.transform(
            X_train_standardized[columnsToEncode]
        )
        X_test_standardized[columnsToEncode] = target_encode_categorial.transform(
            X_test_standardized[columnsToEncode]
        )
    X_train_standardized = scalerX.transform(X_train_standardized)
    X_test_standardized = scalerX.transform(X_test_standardized)

    def getGroupNComponent(group):
        return 1 if len(group) == 2 else 2 if len(group) < 5 else 3

    reducers = []
    for group in correlated_groups:
        if len(group) > 1:
            n_comp = getGroupNComponent(group)
            reducer = (
                umap.UMAP(
                    n_components=n_comp,
                    random_state=seed_restricted,
                    output_metric="euclidean",  # CHECK IF CORRECT NOTE
                )
                if parameters["dimRedOpt"] == "UMAP"
                else PCA(n_components=n_comp, random_state=seed_restricted)
            )
            reducer = reducer.fit(X_standardized[group])
            result = reducer.transform(X_standardized[group])
            reducers.append(reducer)
            
            for i in range(n_comp):
                colName = f"REDUCED-{'-'.join(map(str, group))}-{i}"
                X_train_standardized[colName] = reducer.transform(
                    X_train_standardized[group]
                )[:, i]
                X_test_standardized[colName] = reducer.transform(
                    X_test_standardized[group]
                )[:, i]
            X_train_standardized.drop(columns=group, inplace=True)
            X_test_standardized.drop(columns=group, inplace=True)

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

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-4.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        itertools.repeat(
            float,
            (
                X.shape[1]
                - sum([len(group) for group in correlated_groups])
                + sum([getGroupNComponent(group) for group in correlated_groups])
            ),
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
            )
        elif funParam["type"] == "Terminal":
            fun = [f for f in TERMINALS if f["id"] == funParam["id"]][0]
            pset.addEphemeralConstant(fun["id"], fun["function"], fun["out"])
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

    loss = log_loss if classificationOk else mean_squared_error

    def evaluate(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # For multi-label classification
        if classificationOk:
            y_train_predict = [sigmoid(func(*row)) for row in X_train_list]
        else:  # Regression case
            y_train_predict = [func(*row) for row in X_train_list]
        # Convert predictions and targets to arrays for loss calculation
        y_train_predict = np.array(y_train_predict)
        y_train_array = np.array(y_train_list)
        return (loss(y_train_array, y_train_predict), individual.height)

    toolbox.register("evaluate", evaluate)
    toolbox.register(
        "select",
        [s for s in SELECTION_SET if s["id"] == parameters["selectionMethod"]["id"]][0][
            "function"
        ],
    )
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register(
        "expr_mut", gp.genHalfAndHalf, min_=0, max_=max(4, parameters["treeDepth"] // 3)
    )
    mutations = [
        m
        for m in MUTATION_SET
        if m["id"] in [mut["id"] for mut in parameters["mutationFunction"]]
    ]
    toolbox.register(
        "mutate",
        partial(
            custom_mutation,
            rng=rng,
            mutations=mutations,
            pset=pset,
            expr=toolbox.expr_mut,
        ),
    )

    pop = toolbox.population(n=parameters["popSize"])
    hof = tools.HallOfFame(3)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    eaSimpleWithCallback(
        model.user_id,
        pop,
        toolbox,
        parameters["crossChance"],
        parameters["mutationChance"],
        parameters["genCount"],
        stats,
        halloffame=hof,
    )
    
    predictor = create_prediction_function(
        classificationOk,
        toolbox,
        columnsToEncode,
        correlated_groups,
        target_encode_categorial,
        scalerX,
        scalerY,
        reducers,
        getGroupNComponent,
    )
    
    X_test_predict = X_test.copy()
    
    explainer = dx.Explainer(
    model=hof[0],
    data=X_test_predict,
    y=y_test,
    predict_function=predictor,
    model_type="classification" if classificationOk else "regression",
    label="Genetic Programming Model",
    )
    
    performance = explainer.model_performance(model_type="classification" if classificationOk else "regression")
    fig_performance = performance.plot(show=False).to_html(full_html=True, include_plotlyjs="cdn")
    
    single_explanation = explainer.predict_parts(X_test_predict.iloc[0], type="shap")
    fig_single_explanation = single_explanation.plot(show=False).to_html(full_html=True, include_plotlyjs="cdn")
    
    profile = explainer.model_profile(random_state=seed_restricted)
    fig_profile = profile.plot(show=False).to_html(full_html=True, include_plotlyjs="cdn")
    
    saveModel(
        model,
        hof[0],
        predictor,
        fig_performance,
        fig_single_explanation,
        fig_profile,
    )
    
    return hof[0]
    

def saveModel(
    model,
    best_individual,
    predictor,
    fig_performance,
    fig_single_explanation,
    fig_profile,
):
    try:
        if (
            not model
            or not best_individual
            or not predictor
            or not fig_performance
            or not fig_single_explanation
            or not fig_profile
        ):
            return False
        modelId = str(model.id)

        os.makedirs(f"models/{modelId}", exist_ok=True)

        with open(f"models/{modelId}/model.pkl", "wb") as f:
            dill.dump(best_individual, f)
        
        with open(f"models/{modelId}/predictor.pkl", "wb") as f:
            dill.dump(predictor, f)
            
        with open(f"models/{modelId}/fig_performance.html", "w", encoding="utf-8") as f:
            f.write(fig_performance)
            
        with open(f"models/{modelId}/fig_single_explanation.html", "w", encoding="utf-8") as f:
            f.write(fig_single_explanation)
            
        with open(f"models/{modelId}/fig_profile.html", "w", encoding="utf-8") as f:
            f.write(fig_profile)
            
        # model.setResourcesPath(f"models/{modelId}")


        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


# def predict_function(
#     model,
#     data,
#     toolbox,
#     columnsToEncode,
#     correlated_groups,
#     target_encode_categorial,
#     scalerX,
#     reducers,
#     getGroupNComponent,
# ):
#     cols = []

#     if len(columnsToEncode):
#         data[columnsToEncode] = target_encode_categorial.transform(
#             data[columnsToEncode]
#         )

#     data_standardized = scalerX.transform(data)
#     for index, group  in enumerate(correlated_groups):
#         reducer = reducers[index]
#         n_comp = getGroupNComponent(group)
#         for i in range(n_comp):
#             colName = f"REDUCED-{'-'.join(map(str, group))}-{i}"
#             cols.append(colName)

#             data_standardized[colName] = reducer.transform(data_standardized[group])[
#                 :, i
#             ]
#         data_standardized.drop(columns=group, inplace=True)

#     dataList = data_standardized.values.tolist()
#     func = toolbox.compile(model[0])  # Compile the best individual
#     return np.array([sigmoid(func(*row)) for row in dataList])


def create_prediction_function(
    classificationOk,
    toolbox,
    columnsToEncode,
    correlated_groups,
    target_encode_categorial,
    scalerX,
    scalerY,
    reducers,
    getGroupNComponent,
):
    def predict(model, dataOriginal):
        data = dataOriginal.copy()

        if len(columnsToEncode):
            data[columnsToEncode] = target_encode_categorial.transform(
                data[columnsToEncode]
            )

        data_standardized = scalerX.transform(data)  # Standardize the data

        for index, group in enumerate(correlated_groups):
            reducer = reducers[index]
            n_comp = getGroupNComponent(group)
            for i in range(n_comp):
                colName = f"REDUCED-{'-'.join(map(str, group))}-{i}"
                data_standardized[colName] = reducer.transform(
                    data_standardized[group]
                )[:, i]
            data_standardized.drop(columns=group, inplace=True)

        dataList = data_standardized.values.tolist()
        func = toolbox.compile(model)
        
        if classificationOk:
            return np.array([sigmoid(func(*row)) for row in dataList])
        else:
            predictions = np.array([func(*row) for row in dataList])
            return scalerY.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return predict
