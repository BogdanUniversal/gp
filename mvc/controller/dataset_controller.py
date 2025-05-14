from flask_jwt_extended import get_jwt_identity, jwt_required
import pandas as pd
from mvc.view.dataset_view import createDataset, getDataset, getDatasets
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
            {
                "id": dataset.id,
                "name": dataset.file_name,
                "upload_date": dataset.upload_date,
            }
            for dataset in datasets
        ]
        datasets = sorted(datasets, key=lambda x: x["upload_date"], reverse=True)

        return jsonify({"datasets": datasets}), 200
    except Exception:
        return jsonify({"datasets": "An error occurred"}), 500


@datasetBp.route("/get_dataset", methods=["GET"])
@jwt_required()
def getDatasetRoute():
    try:
        user_id = getUser(get_jwt_identity()).id
        dataset_id = request.args.get("dataset_id")

        if not dataset_id:
            return jsonify({"dataset": "Dataset ID is required"}), 422

        dataset, data, columns = getDataset(user_id, dataset_id)

        if dataset is None:
            return jsonify({"dataset": "Dataset not found"}), 404

        return (
            jsonify(
                {
                    "dataset": {
                        "id": dataset.id,
                        "name": dataset.file_name,
                        "upload_date": dataset.upload_date,
                        "data": data,
                        "columns": columns,
                        "total_rows": len(data),
                        "total_columns": len(columns),
                    }
                }
            ),
            200,
        )
    except Exception:
        return jsonify({"dataset": "An error occurred"}), 500


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
