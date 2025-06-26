from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class PizzaRequestInput(BaseModel):
    """
    Defines the complete set of raw features required to make a prediction.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "t3_w5491",
                "request_title": "Broke college student hoping for a pizza",
                "request_text_edit_aware": "I have an exam tomorrow and haven't eaten all day. A pizza would be a lifesaver!",
                "requester_username": "studious_student",
                "unix_timestamp_of_request_utc": 1380481858.0,
                "requester_account_age_in_days_at_request": 365.5,
                "requester_days_since_first_post_on_raop_at_request": 0.0,
                "requester_number_of_comments_at_request": 25,
                "requester_number_of_comments_in_raop_at_request": 2,
                "requester_number_of_posts_at_request": 10,
                "requester_number_of_posts_on_raop_at_request": 1,
                "requester_number_of_subreddits_at_request": 5,
                "requester_upvotes_minus_downvotes_at_request": 150,
                "requester_upvotes_plus_downvotes_at_request": 250,
                "requester_subreddits_at_request": ["funny", "askreddit", "pics"],
                "unix_timestamp_of_request": 1380481858.0,
            }
        }
    )

    # --- Text and Identifier Features ---
    request_id: str = Field(..., json_schema_extra={"example": "t3_w5491"})
    request_title: str = Field(
        ..., json_schema_extra={"example": "Broke college student hoping for a pizza"}
    )
    request_text_edit_aware: str = Field(
        ..., json_schema_extra={"example": "I have an exam tomorrow..."}
    )
    requester_username: str = Field(
        ..., json_schema_extra={"example": "studious_student"}
    )

    # --- Time Feature ---
    unix_timestamp_of_request_utc: float = Field(
        ..., json_schema_extra={"example": 1380481858.0}
    )

    # --- Numeric Features (Core model inputs) ---
    requester_account_age_in_days_at_request: float = Field(
        ..., json_schema_extra={"example": 365.5}
    )
    requester_days_since_first_post_on_raop_at_request: float = Field(
        ..., json_schema_extra={"example": 0.0}
    )
    requester_number_of_comments_at_request: int = Field(
        ..., json_schema_extra={"example": 25}
    )
    requester_number_of_comments_in_raop_at_request: int = Field(
        ..., json_schema_extra={"example": 2}
    )
    requester_number_of_posts_at_request: int = Field(
        ..., json_schema_extra={"example": 10}
    )
    requester_number_of_posts_on_raop_at_request: int = Field(
        ..., json_schema_extra={"example": 1}
    )
    requester_number_of_subreddits_at_request: int = Field(
        ..., json_schema_extra={"example": 5}
    )
    requester_upvotes_minus_downvotes_at_request: int = Field(
        ..., json_schema_extra={"example": 150}
    )
    requester_upvotes_plus_downvotes_at_request: int = Field(
        ..., json_schema_extra={"example": 250}
    )

    # Features present in raw data that the pipeline expects to drop, optional.
    # We include so  input DataFrame for prediction has the exact same columns as the DataFrame used for training.
    requester_subreddits_at_request: List[str] = Field(
        ..., json_schema_extra={"example": ["funny", "askreddit", "pics"]}
    )
    unix_timestamp_of_request: Optional[float] = Field(
        None, json_schema_extra={"example": 1380481858.0}
    )
