import uuid
import bcrypt
from sqlalchemy.dialects.postgresql import UUID, TEXT
from extensions import db

class User(db.Model):
    __tablename__ = "User"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = db.Column(TEXT, unique=True, nullable=False)
    first_name = db.Column(TEXT, nullable=False)
    last_name = db.Column(TEXT, nullable=False)
    password = db.Column(TEXT, nullable=False)

    def __init__(self, email, firstName, lastName, password):
        self.email = email
        self.first_name = firstName
        self.last_name = lastName
        self.password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verifyPassword(self, password):
        return bcrypt.checkpw(password.encode(), self.password.encode())
    