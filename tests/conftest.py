import os

# Configure the app for an isolated, in-memory test database before
# importing main.py (which creates tables at import time).
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["FLASK_ENV"] = "testing"
os.environ["FLASK_DEBUG"] = "0"

import pytest
from werkzeug.security import generate_password_hash

import main as truthlens


@pytest.fixture()
def app():
    app = truthlens.app
    app.config.update(TESTING=True)

    with app.app_context():
        truthlens.db.drop_all()
        truthlens.db.create_all()

    yield app

    with app.app_context():
        truthlens.db.session.remove()
        truthlens.db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def create_user(role="user", email=None, password="password123",
                is_active=True):
    with truthlens.app.app_context():
        user = truthlens.User(
            name=role.title(),
            email=email or f"{role}@example.com",
            password=generate_password_hash(password),
            role=role,
            is_active=is_active,
        )
        truthlens.db.session.add(user)
        truthlens.db.session.commit()
        return user.id


@pytest.fixture()
def admin_id(app):
    return create_user("admin", "admin@example.com")


@pytest.fixture()
def user_id(app):
    return create_user("user", "user@example.com")


def login_session(client, user_id, role="admin", token="test-csrf"):
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["user_name"] = role.title()
        session["user_email"] = f"{role}@example.com"
        session["user_role"] = role
        session["admin_csrf_token"] = token
