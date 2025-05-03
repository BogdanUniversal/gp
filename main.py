from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from config.config import Config
from mvc.controller.user_controller import userBp
from mvc.controller.dataset_controller import datasetBp
from extensions import db


app = Flask(__name__)
app.config.from_object(Config)

jwt = JWTManager(app)

CORS(app, supports_credentials=True)

db.init_app(app)

# with app.app_context():
#     db.create_all()
migrate = Migrate(app, db)

app.register_blueprint(userBp, url_prefix="/users")
app.register_blueprint(datasetBp, url_prefix="/datasets")

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
    