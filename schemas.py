from pydantic import BaseModel, Field

class PizzaRequestInput(BaseModel):
    request_title: str = Field(..., json_schema_extra={"example": "Broke college student hoping for a pizza"})
    request_text_edit_aware: str = Field(..., json_schema_extra={"example": "I have an exam tomorrow..."})
    unix_timestamp_of_request_utc: float = Field(..., json_schema_extra={"example": 1380481858.0})
    
    requester_account_age_in_days_at_request: float = Field(..., json_schema_extra={"example": 365.5})
    requester_days_since_first_post_on_raop_at_request: float = Field(..., json_schema_extra={"example": 0.0})
    requester_number_of_comments_at_request: int = Field(..., json_schema_extra={"example": 25})
    requester_number_of_comments_in_raop_at_request: int = Field(..., json_schema_extra={"example": 2})
    requester_number_of_posts_at_request: int = Field(..., json_schema_extra={"example": 10})
    requester_number_of_posts_on_raop_at_request: int = Field(..., json_schema_extra={"example": 1})
    requester_number_of_subreddits_at_request: int = Field(..., json_schema_extra={"example": 5})
    requester_upvotes_minus_downvotes_at_request: int = Field(..., json_schema_extra={"example": 150})
    requester_upvotes_plus_downvotes_at_request: int = Field(..., json_schema_extra={"example": 250})