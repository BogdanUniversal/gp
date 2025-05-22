from deap import base, creator, tools
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from general_set import LOSSES_SET
from mvc.model.dataset_cache import dataset_cache
from mvc.model.parameters_cache import parameters_cache


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


def run_genetic_algorithm(user_id):
    dataset = dataset_cache.get(str(user_id)).copy()
    parameters = parameters_cache.get(user_id)

    seed = np.random.SeedSequence()

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
