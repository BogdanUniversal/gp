from flask import Flask
from flask_cors import CORS
from config.config import Config
from mvc.controller.user_controller import bp
from extensions import db
from flask_jwt_extended import JWTManager

app = Flask(__name__)
app.config.from_object(Config)

jwt = JWTManager(app)

CORS(app, supports_credentials=True)

db.init_app(app)

with app.app_context():
    db.create_all()

app.register_blueprint(bp, url_prefix="/users")

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
    