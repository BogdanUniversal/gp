from mvc.model.user import User
from extensions import db


def createAccount(email, firstName, lastName, password):
    user = User.query.filter_by(email=email).first()
    if user:
        return False
    user = User(email, firstName, lastName, password)
    db.session.add(user)
    db.session.commit()
    return True


def verifySignin(email, password):
    user = User.query.filter_by(email=email).first()
    if user:
        return user.verifyPassword(password)
    return False


def getUser(email):
    user = User.query.filter_by(email=email).first()
    return user if user else None