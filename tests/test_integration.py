import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock

os.environ["TESTING"] = "true"

from src.main import app, get_db
from src.database import Base
from src.models import PredictionLog
import src.main  

# test database engine and session factory 
TEST_DB_FILE = "./test_integration.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_FILE}"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


# --db cleanup after entire test session ---
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db_file():
    """
    This fixture runs once after all tests are finished to remove the
    test database file.
    """
    yield
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)


@pytest.fixture(scope="function")
def db_setup_and_teardown():
    """
    This is the core fixture that manages the database state.
    It creates all tables before a test runs and drops them all afterward.
    """
    # vreate all tables defined in your models that use the `Base` declarative base
    Base.metadata.create_all(bind=test_engine)
    yield
    # drop all tables to ensure a clean slate for the next test
    Base.metadata.drop_all(bind=test_engine)


# function that will be injected into the FastAPI app during tests
def override_get_db():
    """A replacement for the `get_db` dependency that uses the test database."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client(db_setup_and_teardown, monkeypatch):
    """
    This fixture sets up the test client. It depends on `db_setup_and_teardown`
    to ensure the database is ready *before* the client is created.
    """
    # Mock the machine learning model to avoid loading the actual file
    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]]
    monkeypatch.setattr(src.main, "model_pipeline", mock_model)

    # Apply the dependency override for the /predict endpoint
    app.dependency_overrides[get_db] = override_get_db

    # Now that everything is patched and the DB is set up, create the client
    with TestClient(app) as test_client:
        yield test_client

    # Clean up the override after the test is done
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def db_session(db_setup_and_teardown):
    """
    This fixture provides a direct session to the test database for asserting
    that data was written correctly. It also depends on the setup fixture.
    """
    db = TestingSessionLocal()
    yield db
    db.close()



def test_valid_prediction(client, db_session):
    """
    GIVEN a valid payload
    WHEN the /predict endpoint is called
    THEN it should return a 200 OK, the correct prediction, and create a DB log
    """
    valid_payload = {
        "request_id": "test_123",
        "request_title": "Valid integration test",
        "request_text_edit_aware": "This is a valid test for the integration.",
        "requester_username": "test_user",
        "unix_timestamp_of_request_utc": 1380000000.0,
        "requester_account_age_in_days_at_request": 100,
        "requester_days_since_first_post_on_raop_at_request": 10,
        "requester_number_of_comments_at_request": 5,
        "requester_number_of_comments_in_raop_at_request": 1,
        "requester_number_of_posts_at_request": 2,
        "requester_number_of_posts_on_raop_at_request": 1,
        "requester_number_of_subreddits_at_request": 3,
        "requester_upvotes_minus_downvotes_at_request": 50,
        "requester_upvotes_plus_downvotes_at_request": 100,
        "requester_subreddits_at_request": ["test", "pizza"],
        "unix_timestamp_of_request": 1380000000.0,
    }

    response = client.post("/predict", json=valid_payload)

    # AssertAPI Response
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] == "Pizza Received"
    assert data["prediction_value"] == 1
    assert data["probability_of_success"] == 0.9

    # Assert Database Log
    log_entry = db_session.query(PredictionLog).first()
    assert log_entry is not None
    assert log_entry.prediction_value == 1
    assert log_entry.raw_request["request_title"] == "Valid integration test"


def test_invalid_prediction_missing_field(client, db_session):
    """
    GIVEN an invalid payload (missing a required field)
    WHEN the /predict endpoint is called
    THEN it should return a 422 Unprocessable Entity and not create a DB log
    """
    invalid_payload = {
        # "request_title" is missing
        "request_text_edit_aware": "This is an invalid test.",
        "unix_timestamp_of_request_utc": 1380000000.0,
        "requester_account_age_in_days_at_request": 100,
        "requester_days_since_first_post_on_raop_at_request": 10,
        "requester_number_of_comments_at_request": 5,
        "requester_number_of_comments_in_raop_at_request": 1,
        "requester_number_of_posts_at_request": 2,
        "requester_number_of_posts_on_raop_at_request": 1,
        "requester_number_of_subreddits_at_request": 3,
        "requester_upvotes_minus_downvotes_at_request": 50,
        "requester_upvotes_plus_downvotes_at_request": 100,
    }

    response = client.post("/predict", json=invalid_payload)

    # assert
    # API Response
    assert response.status_code == 422

    # Database Log
    log_count = db_session.query(PredictionLog).count()
    assert log_count == 0
