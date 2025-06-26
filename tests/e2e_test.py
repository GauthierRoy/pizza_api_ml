# e2e_test.py
import requests
import time
import subprocess
import pytest

# issue, will write into the real database.

simple_valid_test_payload = {
    "request_id": "test_e2e_123",
    "request_title": "Long week of exams, a pizza would be a dream!",
    "request_text_edit_aware": "I have been studying non-stop. Would be so grateful if a kind soul could send a pizza my way. Thank you for considering!",
    "requester_username": "test_student",
    "unix_timestamp_of_request_utc": 1378819200.0,
    "requester_account_age_in_days_at_request": 250.5,
    "requester_days_since_first_post_on_raop_at_request": 0,
    "requester_number_of_comments_at_request": 55,
    "requester_number_of_comments_in_raop_at_request": 0,
    "requester_number_of_posts_at_request": 12,
    "requester_number_of_posts_on_raop_at_request": 1,
    "requester_number_of_subreddits_at_request": 15,
    "requester_upvotes_minus_downvotes_at_request": 450,
    "requester_upvotes_plus_downvotes_at_request": 800,
    "requester_subreddits_at_request": ["funny", "askreddit", "pics"],
    "unix_timestamp_of_request": 1378819200.0,
}

simple_invalid_test_payload = {
    # "request_title": "Long week of exams, a pizza would be a dream!", # Title is missing
    "request_id": "test_e2e_invalid",
    "request_text_edit_aware": "I have been studying non-stop. Would be so grateful if a kind soul could send a pizza my way. Thank you for considering!",
    "requester_username": "test_student",
    "unix_timestamp_of_request_utc": 1378819200.0,
    "requester_account_age_in_days_at_request": 250.5,
    "requester_days_since_first_post_on_raop_at_request": 0,
    "requester_number_of_comments_at_request": 55,
    "requester_number_of_comments_in_raop_at_request": 0,
    "requester_number_of_posts_at_request": 12,
    "requester_number_of_posts_on_raop_at_request": 1,
    "requester_number_of_subreddits_at_request": 15,
    "requester_upvotes_minus_downvotes_at_request": 450,
    "requester_upvotes_plus_downvotes_at_request": 800,
    "requester_subreddits_at_request": ["funny", "askreddit", "pics"],
    "unix_timestamp_of_request": 1378819200.0,
}


@pytest.fixture(scope="session")
def running_service():
    """
    A pytest fixture that starts the Docker containers before any tests run
    and tears them down after all tests are complete.
    'scope="session"' means this runs only ONCE for the entire test run.
    """
    api_url = "http://localhost:8000"

    try:
        # 1. start Docker containers
        print("\n--- [SETUP] Starting Docker services... ---")
        subprocess.run(["docker", "compose", "up", "-d", "--build"], check=True)

        # Wait for the API to be available
        for i in range(15):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{api_url}/docs")
                if response.status_code == 200:
                    print("✅ [SETUP] API is up and running.")
                    break
            except requests.exceptions.ConnectionError:
                pass
            print(f"⏳ [SETUP] Waiting for API... (attempt {i + 1})")
            time.sleep(2)
        else:
            # If the loop finishes without breaking, raise an error
            pytest.fail("API did not become available.")

        yield api_url

    finally:
        # 4. TEARDOWN: This code runs after all tests are done
        print("\n--- [TEARDOWN] Shutting down Docker services... ---")
        subprocess.run(["docker", "compose", "down"])


def test_e2e_valid_call(running_service):
    """
    Tests a valid call to the running service.
    The 'running_service' argument tells pytest to run our fixture first.
    """
    api_url = running_service  # The fixture yields the URL for us to use

    response = requests.post(f"{api_url}/predict", json=simple_valid_test_payload)

    assert response.status_code == 200
    assert "prediction_label" in response.json()


def test_e2e_invalid_call(running_service):
    """
    Tests an invalid call to the running service.
    """
    api_url = running_service

    response = requests.post(f"{api_url}/predict", json=simple_invalid_test_payload)

    assert response.status_code == 422
    assert "detail" in response.json()
