from mvc.model.dataset import Dataset
from extensions import db
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root directory
DATASETS_PATH = os.path.join(BASE_DIR, "datasets")  # Path to the 'datasets' folder

def createDataset(name, description, user_id, data):
    try:
        os.makedirs(os.path.join(DATASETS_PATH, str(user_id)), exist_ok=True)
        file_path = os.path.join(DATASETS_PATH, str(user_id), name + ".csv")
        data.save(file_path)
        
        dataset = Dataset(name, description, user_id)
        
        db.session.add(dataset)
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False