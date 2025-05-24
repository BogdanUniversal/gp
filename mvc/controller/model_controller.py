from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from mvc.view.model_view import getLossFunctions, getTerminalsPrimitives
from mvc.view.user_view import getUser
from mvc.model.parameters_cache import parameters_cache
from genetic_programming.primitive_set_gp import PRIMITIVES
from genetic_programming.terminal_set_gp import TERMINALS
from genetic_programming.general_set import (
    CORRELATION_SET,
    DIMENSIONALITY_REDUCTION_SET,
    LOSSES_SET,
    MUTATION_SET,
    SELECTION_SET,
)

modelBp = Blueprint("model", __name__)


@modelBp.route("/get_loss_functions", methods=["GET"])
@jwt_required()
def getLossFunctionsRoute():
    try:
        args = request.args

        currentUser = get_jwt_identity()
        currentUserId = getUser(currentUser).id

        labelColumn = args.get("label")

        lossFunctions = getLossFunctions(currentUserId, labelColumn)

        return jsonify(
            [{"id": fun.get("id"), "name": fun.get("name")} for fun in lossFunctions]
        )
    except:
        return jsonify({"loss": "No loss functions found!"}), 402


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

        if args["selectedLabel"] == "":
            okParameters = False
        if args["corrOpt"] not in CORRELATION_SET:
            okParameters = False
        if args["lossFunction"]["id"] not in [
            loss["id"] for loss_group in LOSSES_SET for loss in loss_group
        ]:
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

        if okParameters == False:
            return jsonify({"error": "Invalid parameters!"}), 402

        currentUser = get_jwt_identity()
        currentUserId = getUser(currentUser).id

        parameters_cache.set(currentUserId, args)
        print(f"Parameters set for user {currentUserId}: {args}")
        # Here you would typically save these parameters to a database or use them in your application logic
        # For demonstration purposes, we'll just return them in the response

        return jsonify({"parameters": "Parameters set sucessfully!"})
    except:
        return jsonify({"error": "Failed to set parameters!"}), 402
