from datetime import timedelta
import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql+psycopg2://postgres:marelerege@localhost:5432/genetic_programming') # NOTE CHANGE THIS FOR DISTRIBUTION
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = "e37eebca-9a6f-4faa-afb2-3bec6267a4d8" # HIDE THIS FOR DISTRIBUTION
    JWT_COOKIE_SECURE = False # Change to True in production for HTTPS only
    # JWT_CSRF_IN_COOKIES = False
    JWT_COOKIE_CSRF_PROTECT = True
    JWT_TOKEN_LOCATION = ["cookies"]
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    