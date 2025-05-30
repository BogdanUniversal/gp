import uuid
from sqlalchemy.dialects.postgresql import UUID, TEXT
from sqlalchemy import ForeignKey
from extensions import db
from datetime import datetime

class Model(db.Model):
    __tablename__ = "Model"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)
    model_file_name = db.Column(TEXT, nullable=True)
    dim_red_file_name = db.Column(TEXT, nullable=True)
    plots_path = db.Column(TEXT, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, user_id):
        self.user_id = user_id
        self.model_file_name = None
        self.dim_red_file_name = None
        self.plots_path = None 
        
    def setModelFileName(self, file_name):
        self.model_file_name = file_name
        
    def setDimRedFileName(self, file_name):
        self.dim_red_file_name = file_name
        
    def setPlotsPath(self, plots_path):
        self.plots_path = plots_path
        