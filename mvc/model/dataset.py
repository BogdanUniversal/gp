import uuid
from sqlalchemy.dialects.postgresql import UUID, TEXT
from sqlalchemy import ForeignKey
from extensions import db
from datetime import datetime

class Dataset(db.Model):
    __tablename__ = "Dataset"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)  # Foreign key to User
    file_name = db.Column(TEXT, nullable=False)  # Name of the CSV file
    upload_date = db.Column(db.DateTime, default=datetime.now)  # Timestamp of upload
    file_path = db.Column(TEXT, nullable=False)  # Path to the file on the filesystem or cloud

    def __init__(self, user_id, file_name, file_path):
        self.user_id = user_id
        self.file_name = file_name
        self.file_path = file_path