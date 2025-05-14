from datetime import datetime, timedelta, timezone
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt,
    get_jwt_identity,
    set_access_cookies,
)
from flask_migrate import Migrate
from config.config import Config
from mvc.controller.user_controller import userBp
from mvc.controller.dataset_controller import datasetBp
from mvc.controller.model_controller import modelBp
from extensions import db


app = Flask(__name__)
app.config.from_object(Config)

jwt = JWTManager(app)

# CORS(
#     app,
#     supports_credentials=True,
#     origins=["http://localhost:3000"],
#     resources={r"/*": {"origins": "http://localhost:3000"}},
# )

CORS(app, 
     supports_credentials=True,
     resources={
         r"/*": {
             "origins": ["http://localhost:3000"],
             "allow_headers": ["Content-Type", "X-CSRF-TOKEN"],
             "expose_headers": ["Content-Type", "X-CSRF-TOKEN"]
         }
     })

db.init_app(app)

# with app.app_context():
#     db.create_all()
migrate = Migrate(app, db)

app.register_blueprint(userBp, url_prefix="/users")
app.register_blueprint(datasetBp, url_prefix="/datasets")
app.register_blueprint(modelBp, url_prefix="/models")


@app.after_request
def refresh_expiring_jwts(response):
    """
    Automatically refresh the access token if the refresh token is valid and the access token is close to expiring.
    """
    try:
        # Check if the access token is close to expiring
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))

        if target_timestamp > exp_timestamp:
            # Use the refresh token to generate a new access token
            identity = get_jwt_identity()  # Get the identity from the current token
            access_token = create_access_token(
                identity=identity
            )  # Create a new access token
            set_access_cookies(
                response, access_token
            )  # Set the new access token in cookies

    except (RuntimeError, KeyError):
        # If there's no valid access token, check for a valid refresh token
        try:
            # Use the refresh token to generate a new access token
            identity = get_jwt_identity()  # Get the identity from the refresh token
            access_token = create_access_token(
                identity=identity
            )  # Create a new access token
            set_access_cookies(
                response, access_token
            )  # Set the new access token in cookies
        except Exception:
            # If no valid refresh token is present, return the original response
            pass

    return response


# @app.after_request
# def after_request(response):
#     header = response.headers
#     print("-------------------------------asd--------------------------")
#     header['Access-Control-Allow-Origin'] = 'http://localhost:3000'
#     header['Access-Control-Allow-Credentials'] = "true"
#     header['Access-Control-Allow-Methods'] = "GET, POST, PATCH, PUT, DELETE, OPTIONS"
#     header['Access-Control-Allow-Headers'] = "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With"
#     return response

if __name__ == "__main__":
    app.run(debug=True)
