from functools import partial
import threading
from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from mvc.model.dataset_cache import dataset_cache
from mvc.view.model_view import (
    createModel,
    getFeatureTypes,
    getTerminalsPrimitives,
    getModels,
    getPerformance,
    getTree,
    makePrediction
)
from mvc.view.user_view import getUser
from mvc.view.dataset_view import getDataset
from mvc.model.parameters_cache import parameters_cache
from genetic_programming.primitive_set_gp import PRIMITIVES
from genetic_programming.terminal_set_gp import TERMINALS
from genetic_programming.general_set import (
    CORRELATION_SET,
    DIMENSIONALITY_REDUCTION_SET,
    MUTATION_SET,
    SELECTION_SET,
)

modelBp = Blueprint("model", __name__)


@modelBp.route("/get_terminals_primitives", methods=["GET"])
@jwt_required()
def getTerminalsPrimitivesRoute():
    try:
        terminalsPrimitives = getTerminalsPrimitives()

        return jsonify(
            [
                {"id": tp.get("id"), "name": tp.get("name"), "type": tp.get("type")}
                for tp in terminalsPrimitives
            ]
        )
    except:
        return jsonify({"error": "No terminals or primitives found!"}), 402


@modelBp.route("/set_parameters", methods=["POST"])
@jwt_required()
def setParametersRoute():
    try:
        args = request.get_json()

        okParameters = True

        if args["selectedFeatures"] == []:
            okParameters = False
        if args["selectedLabel"] == "":
            okParameters = False
        if args["corrOpt"] not in CORRELATION_SET:
            okParameters = False
        if args["dimRedOpt"] not in DIMENSIONALITY_REDUCTION_SET:
            okParameters = False
        if args["popSize"] <= 0 or args["popSize"] > 1000:
            okParameters = False
        if args["genCount"] <= 0 or args["genCount"] > 500:
            okParameters = False
        if args["treeDepth"] <= 0 or args["treeDepth"] > 25:
            okParameters = False
        if args["crossChance"] < 0.01 or args["crossChance"] > 1:
            okParameters = False
        if args["mutationChance"] < 0.01 or args["mutationChance"] > 1:
            okParameters = False
        if not all(
            [
                (True if x["id"] in [mut["id"] for mut in MUTATION_SET] else False)
                for x in args["mutationFunction"]
            ]
        ):
            okParameters = False
        if args["selectionMethod"]["id"] not in [
            selection["id"] for selection in SELECTION_SET
        ]:
            okParameters = False
        if not all(
            [
                (
                    True
                    if x["id"] in [prim["id"] for prim in TERMINALS + PRIMITIVES]
                    else False
                )
                for x in args["functions"]
            ]
        ):
            okParameters = False
        if args["objective"] not in ["Classification", "Regression"]:
            okParameters = False

        if okParameters == False:
            return jsonify({"error": "Invalid parameters!"}), 402

        currentUser = get_jwt_identity()
        currentUserId = getUser(currentUser).id

        parameters_cache.set(str(currentUserId), args)

        return jsonify({"parameters": "Parameters set sucessfully!"})
    except:
        return jsonify({"error": "Failed to set parameters!"}), 402


@modelBp.route("/train_model", methods=["POST"])
@jwt_required()
def trainModelRoute():
    try:
        currentUser = get_jwt_identity()
        currentUserId = getUser(currentUser).id

        args = request.get_json()
        dataset_id = args.get("dataset_id")
        model_name = args.get("model_name")
        
        if not dataset_id or not model_name:
            return jsonify({"error": "Dataset ID and model name are required!"}), 422

        model = createModel(
            user_id=currentUserId,
            dataset_id=dataset_id,
            model_name=model_name,
        )

        if not model:
            return jsonify({"error": "No parameters/dataset set!"}), 402
        
        if type(model) == IndexError:
            return jsonify({"error": str(model)}), 410

        return jsonify({"message": "Model training started successfully!"})
    except Exception as e:
        if type(e) == IndexError:
            return jsonify({"error": str(e)}), 410
        return jsonify({"error": "Failed to start model training!"}), 402


@modelBp.route("/get_models", methods=["GET"])
@jwt_required()
def getModelsRoute():
    try:
        currentUser = get_jwt_identity()
        currentUserId = getUser(currentUser).id

        models = getModels(currentUserId)

        if not models:
            return jsonify({"error": "No models found!"}), 404

        modelsJson = [
            {
                "id": model.id,
                "dataset_id": model.dataset_id,
                "dataset_name": getDataset(currentUserId, model.dataset_id)[
                    0
                ].file_name,
                "model_name": model.model_name,
                "train_date": model.train_date,
            }
            for model in models
        ]
        modelsJson = sorted(modelsJson, key=lambda x: x["train_date"], reverse=True)

        return jsonify({"models": modelsJson}), 200
    except:
        return jsonify({"error": "Failed to retrieve models!"}), 402


@modelBp.route("/get_performance", methods=["GET"])
@jwt_required()
def getPerformanceRoute():
    try:
        model_id = request.args.get("model_id")

        fig_performance_data, fig_profile_data, fig_single_explanation_data = (
            getPerformance(model_id)
        )
        
        if (
            not fig_performance_data
            or not fig_profile_data
            or not fig_single_explanation_data
        ):
            return jsonify({"error": "Performance data not found!"}), 404

        return (
            jsonify(
                {
                    "fig_performance": fig_performance_data,
                    "fig_profile": fig_profile_data,
                    "fig_single_explanation": fig_single_explanation_data,
                }
            ),
            200,
        )
    except:
        return jsonify({"error": "Failed to retrieve performance data!"}), 402
    
    
@modelBp.route("/get_tree", methods=["GET"])
@jwt_required()
def getTreeRoute():
    try:
        model_id = request.args.get("model_id")
        
        if not model_id:
            return jsonify({"error": "Model ID is required!"}), 422
        
        model_tree = getTree(model_id)
        
        if not model_tree:
            return jsonify({"error": "Model not found!"}), 404
        
        return model_tree, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@modelBp.route("/get_features_types", methods=["GET"])
@jwt_required()
def getFeatureTypesRoute():
    try:
        model_id = request.args.get("model_id")
        if not model_id:
            return jsonify({"error": "Model ID is required!"}), 422
        
        features = getFeatureTypes(model_id)
        if not features:
            return jsonify({"error": "No features found!"}), 404
        
        return jsonify(features), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@modelBp.route("/make_prediction", methods=["POST"])
@jwt_required()
def makePredictionRoute():
    try:
        model_id = request.json.get("model_id")
        data = request.json.get("data")

        if not model_id or not data:
            return jsonify({"error": "Model ID and data are required!"}), 422

        prediction = makePrediction(model_id, data)

        if prediction is None:
            return jsonify({"error": "Prediction failed!"}), 500

        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500