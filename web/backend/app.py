"""All imports"""
from flask import Flask, request
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from pymongo.errors import DuplicateKeyError

from db import save_user, get_user

app = Flask(__name__)
app.secret_key = "granthbagadiagranthbagadia"

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    """Home page"""
    return {
        'success': True
    }


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    """Signup Page"""
    if current_user.is_authenticated:
        return {
            'success': True
        }
    try:
        user_data = request.get_json()
        save_user(user_data)
        return {
            'success': True
        }
    except DuplicateKeyError:
        return {
            'success': False,
            'message': "User already exists!"
        }


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login Page"""
    if current_user.is_authenticated:
        return {
            'success': True
        }
    user_data = request.get_json()
    if user_data['type'] == 'webForm':
        user = get_user(user_data['username'])
        if user and user.check_password(user_data['password']):
            login_user(user)
            return {
                'success': True
            }
        return {
            'success': False,
            'message': "Wrong Password!"
        }
    return {
        'success': False,
        'message': "Wrong Password!"
    }


@app.route("/logout", methods=['GET', 'POST'])
@login_required
def logout():
    """Logout Page"""
    logout_user()
    return {'success': True}



@login_manager.user_loader
def load_user(username):
    """Load User"""
    return get_user(username)



if __name__ == '__main__':
    app.run(debug=True)
