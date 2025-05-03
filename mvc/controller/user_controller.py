from datetime import datetime, timedelta, timezone
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
    set_access_cookies,
    set_refresh_cookies,
    unset_jwt_cookies,
    unset_access_cookies,
    unset_refresh_cookies,
    
)
from mvc.view.user_view import getUser, verifySignin, createAccount

userBp = Blueprint("user", __name__)


@userBp.route("/signin", methods=["POST"])
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


@userBp.route("/signup", methods=["POST"])
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


@userBp.route("/signout", methods=["GET"])
def userSignout():
    response = jsonify({"signout": "Signout successful"})

    unset_jwt_cookies(response)
    return response


@userBp.route("/test", methods=["GET"])
@jwt_required()
def test():
    current_user = get_jwt_identity()
    current_user_name = getUser(current_user).first_name
    print(f"Accessing /test with user: {current_user_name}")
    return jsonify({"test": "test successful", "user": current_user_name})

    
@userBp.after_request
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
            access_token = create_access_token(identity=identity)  # Create a new access token
            set_access_cookies(response, access_token)  # Set the new access token in cookies

    except (RuntimeError, KeyError):
        # If there's no valid access token, check for a valid refresh token
        try:
            # Use the refresh token to generate a new access token
            identity = get_jwt_identity()  # Get the identity from the refresh token
            access_token = create_access_token(identity=identity)  # Create a new access token
            set_access_cookies(response, access_token)  # Set the new access token in cookies
        except Exception:
            # If no valid refresh token is present, return the original response
            pass

    return response