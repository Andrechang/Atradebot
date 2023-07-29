"""Database"""
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

from user import User

client = MongoClient(
    "mongodb+srv://test:test@chatapp.yv6vv.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

chat_db = client.get_database("ChatDB")
users_collection = chat_db.get_collection("users_l2g")


def save_user(user_data):
    """Save User"""
    user = {'_id': user_data['username'], 'email': user_data['email']}
    if user_data['type'] == 'webForm':
        user_data['password'] = generate_password_hash(user_data['password'])
        user['type'] = 'webForm'
        users_collection.insert_one(user)
        return 1
    user['type'] = 'googleForm'
    users_collection.insert_one(user)
    return 1


def get_user(username):
    """User Details"""
    user_data = users_collection.find_one({'_id': username})
    return User(user_data['_id'], user_data['email'], user_data['password']) if user_data else None
