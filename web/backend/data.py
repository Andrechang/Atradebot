"""Data"""
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

from stocks import current_price
from user import User

client = MongoClient(
    "mongodb+srv://test:test@users.qpmwztk.mongodb.net/?retryWrites=true&w=majority")


chat_db = client.get_database("PaperTrading")
users_collection = chat_db.get_collection("users")


def save_user(user_data):
    """Save User"""
    user = {
        '_id': user_data['username'],
        'email': user_data['email'],
        'password': generate_password_hash(user_data['password']),
        'balance': 1000,
        'watchlist': [],
        'portfolio': {},
        'trades': []
    }

    users_collection.insert_one(user)

    return 1


def user_info(username):
    """User Details"""
    user_data = users_collection.find_one({'_id': username})
    user_data['portfolio_value'] = 0
    for symbol in user_data['portfolio']:
        user_data['portfolio_value'] += user_data['portfolio'][symbol]['volume'] * current_price(symbol)[1]
    return user_data if user_data else None


def get_user(username):
    """User Details"""
    user_data = users_collection.find_one({'_id': username})
    return User(user_data['_id'], user_data['email'], user_data['password']) if user_data else None


def change(name, process, symbol, volume, price, time):
    """Change"""
    user_data = user_info(name)

    trade = {
        "symbol": symbol,
        "volume": volume,
        "price": price,
        "type": process,
        "time": time
    }

    try:

        old_volume = user_data["portfolio"][symbol]["volume"]
        old_price = user_data["portfolio"][symbol]["price"]

    except KeyError:

        old_volume = 0
        old_price = 0

    new_balance = user_data['balance']

    if process == 'buy':
        new_volume = old_volume + trade['volume']
        new_price = (old_price*old_volume + trade['price']*trade['volume'])/new_volume
        new_balance -= trade['price']*trade['volume']

    else:
        new_volume = old_volume - trade['volume']
        new_price = (old_price*old_volume - trade['price']*trade['volume'])/new_volume
        new_balance += trade['price']*trade['volume']

    user_data = users_collection.find_one({'_id': name})

    user_data['trades'].append(trade)

    new_data = {
        'trades': user_data['trades'],
        'balance': round(new_balance, 2),
        'portfolio': user_data['portfolio'],
    }

    if new_volume == 0:
        del new_data["portfolio"][symbol]
    else:
        new_data["portfolio"][symbol] = {
            "volume": new_volume,
            "price": new_price
        }

    users_collection.update_one({ '_id': name }, { "$set": new_data })
