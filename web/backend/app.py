"""Main App"""
import json
from flask import (Flask,
                   request,
                   render_template,
                   make_response)

from flask_login import (login_user,
                   LoginManager,
                   login_required,
                   current_user,
                   logout_user)

from data import change, user_info, get_user, save_user
from stocks import current_price

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"


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


@app.route('/about', methods=['GET'])
@login_required
def about():
    """About Page"""
    username = current_user.username
    user = user_info(username)
    return render_template('About.html', user_info=user)


@app.route('/stock/<symbol>', methods=['GET', 'POST'])
def stock_symbol(symbol):
    """Stock Page"""

    if request.method == 'POST':

        username = current_user.username
        volume = int(request.form.get('volume'))
        price = float(request.form.get('price'))
        process = request.form.get('process')
        time = int(request.form.get('time'))

        change(username, process, symbol, volume, price, time)

        return render_template('Stock.html', symbol=symbol)

    return render_template('Stock.html', symbol=symbol)


@app.route('/get_stock/<symbol>', methods=['GET', 'POST'])
def stock(symbol):
    """Get Stock"""

    data = current_price(symbol)
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'

    return response


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
    except: # pylint: disable=bare-except
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
            print(user)
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
