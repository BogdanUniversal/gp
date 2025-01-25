from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, create_refresh_token, get_jwt_identity, jwt_required, set_access_cookies, set_refresh_cookies, unset_jwt_cookies
from mvc.view.user_view import getUser, verifySignin, createAccount

bp = Blueprint("user", __name__)


@bp.route("/signin", methods=["POST"])
def userSignin():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")   

    if not email or not password:
        return jsonify({"signin": "Email and password are required"}), 422
    
    okSignin = verifySignin(email, password)

    if okSignin:
        firstName = getUser(email).first_name
        
        response = jsonify({"signin": "Signin successful", "first_name": firstName})

        access_token = create_access_token(identity=email)
        refresh_token = create_refresh_token(identity=email)

        set_access_cookies(response, access_token)
        set_refresh_cookies(response, refresh_token)

        return response, 201
    return jsonify({"signin": "Wrong credentials!"}), 401


@bp.route("/signup", methods=["POST"])
def userSignup():
    data = request.get_json()

    email = data.get("email")
    firstName = data.get("first_name")
    lastName = data.get("last_name")
    password = data.get("password")   

    if not email or not password or not lastName or not firstName:
        return jsonify({"signup": "All user information is required"}), 422
    
    okSignup = createAccount(email, firstName, lastName, password)

    if okSignup:
        return jsonify({"signup": "User created succesfully!"}), 201
    return jsonify({"signup": "User already exists!"}), 409


@bp.route('/signout', methods=['POST'])
def userSignout():
    response = jsonify({"signout": "Signout successful"})
    unset_jwt_cookies(response)
    return response


@bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    currentUser = get_jwt_identity()
    accessToken = create_access_token(identity=currentUser)

    resp = jsonify({'refresh': True})
    set_access_cookies(resp, accessToken)
    return resp, 200


@bp.route('/test', methods=['GET'])
@jwt_required()
def test():
    print("TEST")
    current_user = get_jwt_identity()
    print(current_user)
    return jsonify({"test": "test successful"})