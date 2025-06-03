from functools import partial
import json
import os
import threading

import pandas as pd
from genetic_programming.gp_algorithm import algorithm
from genetic_programming.primitive_set_gp import *
from genetic_programming.terminal_set_gp import *
from genetic_programming.general_set import *
from mvc.model.parameters_cache import parameters_cache
from mvc.model.dataset_cache import dataset_cache
from mvc.model.model import Model
from extensions import db
import dill
from deap import gp, algorithms, creator, base, tools
from sklearn.preprocessing import LabelEncoder
from flask import current_app


def getTerminalsPrimitives():
    return PRIMITIVES + TERMINALS


def getModels(user_id):
    try:
        models = Model.query.filter_by(user_id=user_id).all()
        return models
    except Exception:
        return None


def createModel(user_id, dataset_id, model_name):
    try:
        parameters = parameters_cache.get(str(user_id))
        datasetCached = dataset_cache.get(str(user_id))

        if (
            not parameters
            or not dataset_id
            or not model_name
            or (datasetCached is None)
        ):
            return None

        feature_types = [
            {
                "name": feature,
                "type": (
                    "number"
                    if pd.api.types.is_numeric_dtype(datasetCached[feature].dtype)
                    else "text"
                ),
            }
            for feature in parameters["selectedFeatures"]
            if feature in datasetCached.columns
        ]

        parameters["featureTypes"] = feature_types
        parameters["labelType"] = {
            "name": parameters["selectedLabel"],
            "type": (
                "number"
                if pd.api.types.is_numeric_dtype(
                    datasetCached[parameters["selectedLabel"]].dtype
                )
                else "text"
            ),
        }

        if parameters["objective"] == "Classification":
            le = LabelEncoder().fit(datasetCached[parameters["selectedLabel"]])
            parameters["labelClasses"] = le.classes_.tolist()

        model = Model(
            user_id=user_id,
            dataset_id=dataset_id,
            model_name=model_name,
            parameters=parameters,
        )
        
        db.session.add(model)
        db.session.commit()
        
        training_thread = threading.Thread(
            target=partial(algorithm, model.id, model.user_id, model.model_name, parameters, datasetCached)
        )

        training_thread.daemon = True
        training_thread.start()
        training_thread.join() 


        model.setResourcesPath(f"models/{model.id}")

        db.session.add(model)
        db.session.commit()

        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        db.session.rollback()
        return None


def getPerformance(model_id):
    try:
        model = Model.query.filter_by(id=model_id).first()

        if not model or not model.resources_path:
            return None

        fig_performance = os.path.join(model.resources_path, "fig_performance.html")
        fig_profile = os.path.join(model.resources_path, "fig_profile.html")
        fig_single_explanation = os.path.join(
            model.resources_path, "fig_single_explanation.html"
        )

        if (
            not os.path.exists(fig_performance)
            or not os.path.exists(fig_profile)
            or not os.path.exists(fig_single_explanation)
        ):
            return None

        with open(fig_performance, "r") as file:
            fig_performance_data = file.read()

        with open(fig_profile, "r") as file:
            fig_profile_data = file.read()

        with open(fig_single_explanation, "r") as file:
            fig_single_explanation_data = file.read()

        return fig_performance_data, fig_profile_data, fig_single_explanation_data
    except Exception:
        return None


def getTree(model_id):
    try:
        model = Model.query.filter_by(id=model_id).first()

        if not model or not model.resources_path:
            return None

        tree_path = os.path.join(model.resources_path, "model_tree.json")

        if not os.path.exists(tree_path):
            return None

        with open(tree_path, "r") as file:
            tree_data = json.load(file)

        return tree_data
    except Exception:
        return None


def getFeatureTypes(model_id):
    try:
        model = Model.query.filter_by(id=model_id).first()

        if not model or not model.parameters:
            return None

        parameters = model.parameters
        feature_types = parameters["featureTypes"]
        label_type = parameters["labelType"]

        if not feature_types or not label_type:
            return None

        return {
            "featureTypes": feature_types,
            "labelType": label_type,
        }
    except Exception:
        return None


def makePrediction(model_id, data):
    try:
        model = Model.query.filter_by(id=model_id).first()

        if not model or not model.resources_path:
            return None

        if not model.parameters or not model.parameters["featureTypes"]:
            return None

        feature_types = {
            item["name"]: item["type"] for item in model.parameters["featureTypes"]
        }

        predict_function_path = os.path.join(model.resources_path, "predictor.pkl")
        if not os.path.exists(predict_function_path):
            return None

        model_path = os.path.join(model.resources_path, "model.pkl")
        if not os.path.exists(model_path):
            return None

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-4.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        with open(predict_function_path, "rb") as file:
            predict_function = dill.load(file)

        with open(model_path, "rb") as file:
            predictor = dill.load(file)

        input_df = pd.DataFrame(data, index=[0])

        for col, dtype in feature_types.items():
            if dtype == "number":
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
            elif dtype == "text":
                input_df[col] = input_df[col].astype(str)
                
        print(f"dataframe dtypes: {input_df.dtypes}")

        prediction = predict_function(predictor, input_df)
        print(f"Prediction VIEW: {prediction[0]}")
        print(f'Prediction VIEW classes: {model.parameters["labelClasses"]}')

        return (
            {
                "prediction": prediction[0],
                "label_classes": model.parameters["labelClasses"],
            }
            if "labelClasses" in model.parameters
            else {"prediction": prediction[0]}
        )
    except Exception as e:
        return None
