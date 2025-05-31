import uuid
from sqlalchemy.dialects.postgresql import UUID, TEXT
from sqlalchemy import ForeignKey
from extensions import db
from datetime import datetime
from sqlalchemy.orm import relationship

class Dataset(db.Model):
    __tablename__ = "Dataset"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)
    file_name = db.Column(TEXT, nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.now) 
    file_path = db.Column(TEXT, nullable=False)
    models = relationship("Model", backref="dataset", cascade="all, delete-orphan")

    def __init__(self, user_id, file_name, file_path):
        self.user_id = user_id
        self.file_name = file_name
        self.file_path = file_path