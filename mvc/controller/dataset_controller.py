from flask_jwt_extended import jwt_required
from mvc.view.dataset_view import createDataset
from flask import Blueprint, request, jsonify

datasetBp = Blueprint("dataset", __name__)

@datasetBp.route("/create", methods=["POST"])
@jwt_required()
def createDatasetRoute():
    data = request.get_json()

    name = data.get("name")
    description = data.get("description")
    user_id = data.get("user_id")
    
    data = data.files["file"]

    if not name or not description or not user_id or not data:
        return jsonify({"create": "All dataset information is required"}), 422

    okCreate = createDataset(name, description, user_id, data)

    if okCreate:
        return jsonify({"create": "Dataset created successfully"}), 201
    return jsonify({"create": "Failed to create dataset"}), 500