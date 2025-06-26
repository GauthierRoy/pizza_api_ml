# feature_engineering.py
import pandas as pd
from typing import Dict, List


def create_features_for_model(raw_data: Dict) -> pd.DataFrame:
    """
    Takes a dictionary of raw request data and performs all necessary
    feature engineering to create the input DataFrame for the model's preprocessor.
    """

    # a. Combine text fields (Corrected to include username)
    # Using .get() with a default empty string is safer for optional fields.
    title = raw_data.get("request_title", "")
    text = raw_data.get("request_text_edit_aware", "")
    username = raw_data.get("requester_username", "")
    full_request_text = f"{title} {text} {username}".strip()

    # b. Calculate text length (Corrected to use only request_text_edit_aware)
    request_length = len(text)

    # c. Extract time-based features
    request_datetime = pd.to_datetime(
        raw_data["unix_timestamp_of_request_utc"], unit="s"
    )
    hour_of_request = request_datetime.hour
    day_of_week = request_datetime.dayofweek  # Monday=0, Sunday=6

    # d. Calculate post ratio
    # Add a small epsilon to avoid division by zero
    denominator = raw_data["requester_number_of_posts_at_request"] + 1e-6
    raop_post_ratio = (
        raw_data["requester_number_of_posts_on_raop_at_request"] / denominator
    )

    # --- Construct the DataFrame for the model ---
    # The dictionary keys and column order must match what the model was trained on.
    feature_dict = {
        # Numeric Features
        "requester_account_age_in_days_at_request": [
            raw_data["requester_account_age_in_days_at_request"]
        ],
        "requester_days_since_first_post_on_raop_at_request": [
            raw_data["requester_days_since_first_post_on_raop_at_request"]
        ],
        "requester_number_of_comments_at_request": [
            raw_data["requester_number_of_comments_at_request"]
        ],
        "requester_number_of_comments_in_raop_at_request": [
            raw_data["requester_number_of_comments_in_raop_at_request"]
        ],
        "requester_number_of_posts_at_request": [
            raw_data["requester_number_of_posts_at_request"]
        ],
        "requester_number_of_posts_on_raop_at_request": [
            raw_data["requester_number_of_posts_on_raop_at_request"]
        ],
        "requester_number_of_subreddits_at_request": [
            raw_data["requester_number_of_subreddits_at_request"]
        ],
        "requester_upvotes_minus_downvotes_at_request": [
            raw_data["requester_upvotes_minus_downvotes_at_request"]
        ],
        "requester_upvotes_plus_downvotes_at_request": [
            raw_data["requester_upvotes_plus_downvotes_at_request"]
        ],
        "request_length": [request_length],
        "raop_post_ratio": [raop_post_ratio],
        # Categorical Features
        "hour_of_request": [hour_of_request],
        "day_of_week": [day_of_week],
        # Text Feature
        "full_request_text": [full_request_text],
    }

    # --- Enforce Column Order ---
    # This is a critical step to ensure the DataFrame structure matches
    # the preprocessor's expectations exactly.
    final_ordered_columns: List[str] = [
        # Numeric
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "request_length",
        "raop_post_ratio",
        # Categorical
        "hour_of_request",
        "day_of_week",
        # Text
        "full_request_text",
    ]

    df = pd.DataFrame(feature_dict)

    return df[final_ordered_columns]  # Return DataFrame with guaranteed column order


# import pandas as pd
# from typing import Dict

# def create_features_for_model(raw_data: Dict) -> pd.DataFrame:
#     """
#     Takes a dictionary of raw request data and performs all necessary
#     feature engineering to create the input DataFrame for the model.

#     Args:
#         raw_data: A dictionary containing the raw input fields.

#     Returns:
#         A pandas DataFrame with all the features ready for the model's preprocessor.
#     """
#     # --- Feature Engineering ---
#     # a. Combine text fields
#     full_request_text = raw_data['request_title'] + " " + raw_data['request_text_edit_aware']

#     # b. Calculate text length
#     request_length = len(full_request_text)

#     # c. Extract time-based features
#     request_datetime = pd.to_datetime(raw_data['unix_timestamp_of_request_utc'], unit='s')
#     hour_of_request = request_datetime.hour
#     day_of_week = request_datetime.dayofweek # Monday=0, Sunday=6

#     # d. Calculate post ratio
#     raop_post_ratio = raw_data['requester_number_of_posts_on_raop_at_request'] / (raw_data['requester_number_of_posts_at_request'] + 1e-6)

#     # --- Construct the DataFrame for the model ---
#     # The dictionary keys and column order must match what the model was trained on.
#     feature_dict = {
#         # Numeric Features
#         'requester_account_age_in_days_at_request': [raw_data['requester_account_age_in_days_at_request']],
#         'requester_days_since_first_post_on_raop_at_request': [raw_data['requester_days_since_first_post_on_raop_at_request']],
#         'requester_number_of_comments_at_request': [raw_data['requester_number_of_comments_at_request']],
#         'requester_number_of_comments_in_raop_at_request': [raw_data['requester_number_of_comments_in_raop_at_request']],
#         'requester_number_of_posts_at_request': [raw_data['requester_number_of_posts_at_request']],
#         'requester_number_of_posts_on_raop_at_request': [raw_data['requester_number_of_posts_on_raop_at_request']],
#         'requester_number_of_subreddits_at_request': [raw_data['requester_number_of_subreddits_at_request']],
#         'requester_upvotes_minus_downvotes_at_request': [raw_data['requester_upvotes_minus_downvotes_at_request']],
#         'requester_upvotes_plus_downvotes_at_request': [raw_data['requester_upvotes_plus_downvotes_at_request']],
#         'request_length': [request_length],
#         'raop_post_ratio': [raop_post_ratio],

#         # Categorical Features
#         'hour_of_request': [hour_of_request],
#         'day_of_week': [day_of_week],

#         # Text Feature
#         'full_request_text': [full_request_text]
#     }

#     return pd.DataFrame(feature_dict)
