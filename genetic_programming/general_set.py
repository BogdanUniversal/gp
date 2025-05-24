from functools import partial
from sklearn.metrics import (
    log_loss,
    hinge_loss,
    mean_squared_error,
    mean_absolute_error,
)
from deap.gp import (
    mutUniform,
    mutEphemeral,
    mutSemantic,
    mutShrink,
    mutNodeReplacement,
    mutInsert,
)
from deap.tools import (
    selTournament,
    selRoulette,
    selBest,
    selNSGA2,
    selNSGA3,
    selSPEA2,
    selLexicase,
)

LOSSES_SET = [
    [
        {"id": "bce", "name": "Binary Cross Entropy", "function": log_loss},
        {"id": "hinge", "name": "Hinge loss", "function": hinge_loss},
    ],
    [{"id": "cce", "name": "Categorical Cross Entropy", "function": log_loss}],
    [
        {"id": "mse", "name": "Mean Squared Error", "function": mean_squared_error},
        {"id": "mae", "name": "Mean Absolute Error", "function": mean_absolute_error},
    ],
]

CORRELATION_SET = ["Pearson", "Spearman", "Kendall"]

DIMENSIONALITY_REDUCTION_SET = ["PCA", "UMAP"]

MUTATION_SET = [
    {"id": "mutUniform", "name": "Uniform Mutation", "function": mutUniform},
    {"id": "mutEphemeral", "name": "Ephemerals Mutation", "function": mutEphemeral},
    {"id": "mutSemantic", "name": "Semantic Mutation", "function": mutSemantic},
    {"id": "mutShrink", "name": "Shrink Mutation", "function": mutShrink},
    {
        "id": "mutNodeReplacement",
        "name": "Node Replacement",
        "function": mutNodeReplacement,
    },
    {"id": "mutInsert", "name": "Insert Mutation", "function": mutInsert},
]

SELECTION_SET = [
        {"id": "tournament", "name": "Tournament Selection", "function": partial(selTournament, tournsize=5)},
        {"id": "roulette", "name": "Roulette Selection", "function": selRoulette},
        {"id": "best", "name": "Best Selection", "function": selBest},
        {"id": "nsga2", "name": "NSGA-II Selection", "function": selNSGA2},
        {"id": "nsga3", "name": "NSGA-III Selection", "function": selNSGA3},
        {"id": "spea2", "name": "SPEA-II Selection", "function": selSPEA2},
        {"id": "lexicase", "name": "Lexicase Selection", "function": selLexicase},
]
