import pandas as pd
from sklearn.metrics import (
    log_loss,
    hinge_loss,
    mean_squared_error,
    mean_absolute_error,
)

from mvc.model.dataset_cache import dataset_cache

BINARY_CLASSIFICATION_LOSS = [
    {"id": "bce", "name": "Binary Cross Entropy", "function": log_loss},
    {"id": "hinge", "name": "Hinge loss", "function": hinge_loss},
]

MULTICLASS_CLASSIFICATION_LOSS = [
    {"id": "cce", "name": "Categorical Cross Entropy", "function": log_loss}
]

REGRESSION_LOSS = [
    {"id": "mse", "name": "Mean Squared Error", "function": mean_squared_error},
    {"id": "mae", "name": "Mean Absolute Error", "function": mean_absolute_error},
]


def getLossFunctions(userId, labelColumn):
    dataset = dataset_cache.get(str(userId))
    labelData = dataset[labelColumn]
    
    isNumeric = pd.api.types.is_numeric_dtype(labelData)
    
    uniqueValues = labelData.nunique()
    
    if isNumeric:
        if uniqueValues > 20:
            return REGRESSION_LOSS
    
    if uniqueValues == 2:
        return BINARY_CLASSIFICATION_LOSS
    elif uniqueValues > 2:
        return MULTICLASS_CLASSIFICATION_LOSS
    else:
        return REGRESSION_LOSS
