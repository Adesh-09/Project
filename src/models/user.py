from flask_sqlalchemy import SQLAlchemy

# SQLAlchemy database instance
# This is initialized in main.py via db.init_app(app)
db = SQLAlchemy()

class User(db.Model):
    """Basic user model."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self) -> str:
        return f"<User {self.username}>"
