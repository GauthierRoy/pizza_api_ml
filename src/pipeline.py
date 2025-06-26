# src/pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer


def drop_leakage_and_redundant_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    leakage_cols = [
        "giver_username_if_known",
        "requester_user_flair",
        "post_was_edited",
    ]
    retrieval_cols = [col for col in df.columns if "_at_retrieval" in col]
    redundant_cols = [
        "request_text",
        "unix_timestamp_of_request",
        "request_id",
        "requester_subreddits_at_request",
    ]
    cols_to_drop = leakage_cols + retrieval_cols + redundant_cols
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True, errors="ignore")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    request_datetime = pd.to_datetime(df["unix_timestamp_of_request_utc"], unit="s")
    df["hour_of_request"] = request_datetime.dt.hour
    df["day_of_week"] = request_datetime.dt.dayofweek
    df.drop(columns=["unix_timestamp_of_request_utc"], inplace=True)
    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["request_length"] = df["request_text_edit_aware"].str.len().fillna(0)
    df["raop_post_ratio"] = df["requester_number_of_posts_on_raop_at_request"] / (
        df["requester_number_of_posts_at_request"] + 1e-6
    )
    df["full_request_text"] = (
        df["request_title"].fillna("")
        + " "
        + df["request_text_edit_aware"].fillna("")
        + " "
        + df["requester_username"].fillna("")
    )
    # Drop original cols that are now part of full_request_text
    df.drop(
        columns=["request_title", "request_text_edit_aware", "requester_username"],
        inplace=True,
        errors="ignore",
    )
    return df



def build_pipeline() -> Pipeline:
    """Builds the full data processing and modeling pipeline."""

    numeric_features = [
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
    ]
    categorical_features = ["hour_of_request", "day_of_week"]
    text_feature = "full_request_text"

    final_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            (
                "text",
                TfidfVectorizer(max_features=1000, stop_words="english"),
                text_feature,
            ),
        ],
        remainder="passthrough",  # Use passthrough to debug if columns are missed
    )

    master_pipeline = Pipeline(
        steps=[
            ("1_drop_cols", FunctionTransformer(drop_leakage_and_redundant_cols)),
            ("2_time_features", FunctionTransformer(create_time_features)),
            ("3_eng_features", FunctionTransformer(create_engineered_features)),
            ("4_preprocessor", final_preprocessor),
        ]
    )

    return master_pipeline
