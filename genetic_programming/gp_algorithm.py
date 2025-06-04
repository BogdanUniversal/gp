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
import inspect
import os
import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
)


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


def update_callback(userId, gen, stats, message):
    update_data = {
        "generation": gen,
        "best_fitness": round(float(stats["min"]), 4),
        "avg_fitness": round(float(stats["avg"]), 4),
        "max_fitness": round(float(stats["max"]), 4),
        "std_dev": round(float(stats["std"]), 4),
        "message": message,
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
    verbose=False,
    stop_threshold=50,
):
    """This algorithm reproduces the simplest evolutionary algorithm with real-time callback updates.
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

    if callback:
        update_callback(userId, 0, record, "initialization")

    best_fitness_so_far = record.get("min", float("inf"))
    stale_generations = 0

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))

        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if callback:
            update_callback(userId, gen, record, "training")

        current_min = record.get("min", float("inf"))
        if current_min < best_fitness_so_far:
            best_fitness_so_far = current_min
            stale_generations = 0
        else:
            stale_generations += 1

        if stale_generations >= stop_threshold:
            break

    if callback:
        update_callback(userId, gen, record, "complete")

    return population, logbook, False if stale_generations >= stop_threshold else True


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def algorithm(
    model_id: str,
    user_id: str,
    model_name: str,
    parametersCached: dict,
    datasetCached: pd.DataFrame,
):
    dataset = datasetCached.copy()
    parameters = parametersCached

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

    def sanitize_column_name(name):
        name = name.replace("-", "_")
        import re

        name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        return name

    input_columns = []
    for col in X_train_standardized.columns:
        input_columns.append(col)

    rename_dict = {}
    for i, col_name in enumerate(input_columns):
        safe_name = "IN_" + sanitize_column_name(col_name)
        rename_dict[f"IN{i}"] = safe_name

    pset.renameArguments(**rename_dict)

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
        func = toolbox.compile(expr=individual)
        if classificationOk:
            y_train_predict = [sigmoid(func(*row)) for row in X_train_list]
        else:
            y_train_predict = [func(*row) for row in X_train_list]

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

    popAlg, log, stop = eaSimpleWithCallback(
        user_id,
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
        label=model_name,
    )

    performance = explainer.model_performance(
        model_type="classification" if classificationOk else "regression"
    )
    fig_performance = performance.plot(show=False).to_html(
        full_html=False, include_plotlyjs=False
    )

    single_explanation = explainer.predict_parts(
        X_test_predict.iloc[0], type="shap", random_state=seed_restricted
    )
    fig_single_explanation = single_explanation.plot(show=False).to_html(
        full_html=False, include_plotlyjs=False
    )

    profile = explainer.model_profile(random_state=seed_restricted, verbose=False)
    fig_profile = profile.plot(show=False).to_html(
        full_html=False, include_plotlyjs=False
    )

    model_tree = tree_to_dict(hof[0], pset)

    saveModel(
        model_id,
        hof[0],
        predictor,
        fig_performance,
        fig_single_explanation,
        fig_profile,
        model_tree,
    )

    return hof[0]


def saveModel(
    model_id,
    best_individual,
    predictor,
    fig_performance,
    fig_single_explanation,
    fig_profile,
    model_tree,
):
    try:
        if (
            not model_id
            or not best_individual
            or not predictor
            or not fig_performance
            or not fig_single_explanation
            or not fig_profile
            or not model_tree
        ):
            return False
        modelId = str(model_id)

        os.makedirs(f"models/{modelId}", exist_ok=True)

        with open(f"models/{modelId}/model.pkl", "wb") as f:
            dill.dump(best_individual, f)

        with open(f"models/{modelId}/predictor.pkl", "wb") as f:
            dill.dump(predictor, f)

        with open(f"models/{modelId}/fig_performance.html", "w", encoding="utf-8") as f:
            f.write(fig_performance)

        with open(
            f"models/{modelId}/fig_single_explanation.html", "w", encoding="utf-8"
        ) as f:
            f.write(fig_single_explanation)

        with open(f"models/{modelId}/fig_profile.html", "w", encoding="utf-8") as f:
            f.write(fig_profile)

        with open(f"models/{modelId}/model_tree.json", "w", encoding="utf-8") as f:
            json.dump(model_tree, f, indent=4)

        return True
    except Exception as e:
        return False


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

        data_standardized = scalerX.transform(data)

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


def tree_to_dict(individual, pset):
    expr = individual

    def format_function_doc(func, in_types=None, out_type=None):
        try:
            name = func.func.__name__ if isinstance(func, partial) else func.__name__

            if in_types:
                params = []
                for i, param_type in enumerate(in_types):
                    type_name = (
                        str(param_type).replace("<class '", "").replace("'>", "")
                    )
                    params.append(f"arg{i}: {type_name}")

                param_str = ", ".join(params)

                return_type = str(out_type).replace("<class '", "").replace("'>", "")

                formatted_sig = f"def {name}({param_str}) -> {return_type}"
            else:
                sig = inspect.signature(func)
                formatted_sig = f"def {name}{str(sig)}"

            doc = func.func.__doc__ if isinstance(func, partial) else func.__doc__
            doc = doc.strip()

            return f"(function) {formatted_sig}\n{doc}"
        except (ValueError, TypeError):
            return func.func.__doc__ if isinstance(func, partial) else func.__doc__

    def _convert_expr(expr, start=0):
        if isinstance(expr[start], gp.Primitive):
            primitive = expr[start]
            prim = [p for p in PRIMITIVES if p["id"] == primitive.name][0]
            result = {
                "name": prim["name"],
                "attributes": {
                    "type": "primitive",
                    "arity": primitive.arity,
                    "doc": format_function_doc(
                        prim["function"], prim["in"], prim["out"]
                    ),
                    "returnType": str(primitive.ret)
                    .replace("<class '", "")
                    .replace("'>", ""),
                },
                "children": [],
            }

            end = start + 1
            for _ in range(primitive.arity):
                child, end = _convert_expr(expr, end)
                result["children"].append(child)

            return result, end

        else:
            terminal = expr[start]
            if hasattr(terminal, "name") and terminal.name.startswith("IN"):
                display_name = terminal.name
                index = int(terminal.name[2:])
                if index < len(pset.arguments):
                    display_name = pset.arguments[index]

                result = {
                    "name": display_name,
                    "attributes": {
                        "type": "variable",
                    },
                }
            elif "rand_" in str(terminal) or isinstance(terminal, (float, int, bool)):
                name = str(terminal)
                nameT = name.split("object")[0].split(".")[-1].strip()

                term = [t for t in TERMINALS if t["id"] == nameT][0]
                result = {
                    "name": term["name"],
                    "attributes": {
                        "type": "ephemeral",
                        "arity": 0,
                        "returnType": str(term["out"])
                        .replace("<class '", "")
                        .replace("'>", ""),
                        "doc": format_function_doc(term["function"], term["out"]),
                        "value": terminal.value,
                    },
                }
            else:
                result = {
                    "name": str(terminal.value),
                    "attributes": {"type": "constant"},
                }

            return result, start + 1

    tree_dict, _ = _convert_expr(expr)
    return tree_dict
