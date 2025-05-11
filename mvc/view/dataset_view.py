from mvc.model.dataset import Dataset
from extensions import db
import os
import uuid


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root directory
DATASETS_PATH = os.path.join(BASE_DIR, "datasets")  # Path to the 'datasets' folder


def getDatasets(user_id):
    try:
        datasets = Dataset.query.filter_by(user_id=user_id).all()
        return datasets
    except Exception:
        return None
    
    
def getDataset(user_id, dataset_id):
    try:
        dataset = Dataset.query.filter_by(user_id=user_id, id=dataset_id).first()
        return dataset
    except Exception:
        return None


def createDataset(user_id, name, data):
    try:
        os.makedirs(os.path.join(DATASETS_PATH, str(user_id)), exist_ok=True)
        file_path = os.path.join(DATASETS_PATH, str(user_id), str(uuid.uuid4()) + ".csv")
        data.save(file_path)
        
        dataset = Dataset(user_id, name, file_path)
        
        db.session.add(dataset)
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False