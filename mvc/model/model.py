import uuid
from sqlalchemy.dialects.postgresql import UUID, TEXT, JSONB
from sqlalchemy import ForeignKey
from extensions import db
from datetime import datetime

class Model(db.Model):
    __tablename__ = "Model"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)
    dataset_id = db.Column(UUID(as_uuid=True), ForeignKey("Dataset.id"), nullable=False)
    model_name = db.Column(TEXT, nullable=False)
    parameters = db.Column(JSONB, nullable=False)
    resources_path = db.Column(TEXT, nullable=True)
    train_date = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, user_id, dataset_id, model_name, parameters):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.model_name = model_name
        self.parameters = parameters
        self.resources_path = None

    def setResourcesPath(self, path):
        self.resources_path = path
        
        