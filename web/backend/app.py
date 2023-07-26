"""Returns all blogs"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "granthbagadiagranthbagadia"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'

db = SQLAlchemy(app)


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    """Home page"""
    return {
        'Name': "Granth",
        "Age": "18",
        "programming": "Learn2Grow"
    }


if __name__ == '__main__':
    app.run(debug=True)
