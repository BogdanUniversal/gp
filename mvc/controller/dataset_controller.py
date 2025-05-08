from flask_jwt_extended import get_jwt_identity, jwt_required
from mvc.view.dataset_view import createDataset, getDatasets
from flask import Blueprint, request, jsonify

from mvc.view.user_view import getUser

datasetBp = Blueprint("dataset", __name__)


@datasetBp.route("/get_datasets", methods=["GET"])
@jwt_required()
def getDatasetsRoute():
    try:
        user_id = getUser(get_jwt_identity()).id
        datasets = getDatasets(user_id)

        if datasets is None:
            return jsonify({"datasets": []}), 200

        datasets = [
            {"name": dataset.file_name, "upload_date": dataset.upload_date}
            for dataset in datasets
        ]
        datasets = sorted(datasets, key=lambda x: x["upload_date"], reverse=True)

        return jsonify({"datasets": datasets}), 200
    except Exception:
        return jsonify({"datasets": "An error occurred"}), 500


@datasetBp.route("/upload", methods=["POST"])
@jwt_required()
def createDatasetRoute():
    try:
        user_id = getUser(get_jwt_identity()).id
        name = request.form.get("name")
        data = request.files["dataset"]

        if not name or not user_id or not data:
            return jsonify({"create": "All dataset information is required"}), 422

        okCreate = createDataset(user_id, name, data)

        if okCreate:
            return jsonify({"create": "Dataset created successfully"}), 201
        return jsonify({"create": "Failed to create dataset"}), 500
    except Exception:
        return jsonify({"create": "An error occurred"}), 500
