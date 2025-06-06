import pandas as pd
from mvc.model.dataset import Dataset
from extensions import db
import os
import uuid

from mvc.model.dataset_cache import dataset_cache


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
        
        df = pd.read_csv(dataset.file_path)
        data = df.head(200).to_dict(orient="records")
        columns = df.columns.tolist()
        
        dataset_cache.set(str(user_id), df)
        
        return dataset, data, columns
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