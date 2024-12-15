from app import db, login
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import UserMixin

class AppUser(UserMixin, db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True)
    password_hash = db.Column(db.String(256))
    chats = db.relationship('Chat', backref='owner', lazy=True)

    def __repr__(self):
        return '<AppUser {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login.user_loader
def load_user(id):
    return AppUser.query.get(int(id))