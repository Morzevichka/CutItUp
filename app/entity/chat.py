from app import db

class Chat(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('app_user.id'), nullable=False)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return '<Chat {}>'.format(self.name)