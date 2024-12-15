from app import db
from datetime import datetime

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    chat_id = db.Column(db.String(36), db.ForeignKey('chat.id', ondelete='CASCADE'), nullable=False)
    sender_type = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<Message {self.filename}>'