from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from mvc.view.model_view import getLossFunctions
from mvc.view.user_view import getUser

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
        
        return jsonify([{"id": fun.get("id"), "name": fun.get("name")} for fun in lossFunctions])
    except:
        return jsonify({"loss": "No loss functions found!"}), 402
        