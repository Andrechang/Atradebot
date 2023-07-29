"""Test Page"""
from werkzeug.security import check_password_hash

class User:
    """Test Page"""
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    @staticmethod
    def is_authenticated():
        """Test Page"""
        return True

    @staticmethod
    def is_active():
        """Test Page"""
        return True

    @staticmethod
    def is_anonymous():
        """Test Page"""
        return False

    def get_id(self):
        """Test Page"""
        return self.username

    def check_password(self, password_input):
        """Test Page"""
        return check_password_hash(self.password, password_input)
