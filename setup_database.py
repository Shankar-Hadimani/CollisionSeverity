from flask_sqlalchemy import SQLAlchemy
from your_application import create_app

app = create_app()
db = SQLAlchemy(app)

def setup_database():
    """Create all database tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created.")

if __name__ == '__main__':
    setup_database() 