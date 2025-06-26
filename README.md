# Pizza Request Success Predictor

A machine learning service that predicts whether a Reddit "Random Acts of Pizza" request will be fulfilled, built as part of a take-home challenge for CybelAngel's machine learning engineer position.

## 📋 Project Overview

This project analyzes 4,040 pizza requests from the Reddit community "Random Acts of Pizza" (December 2010 - September 2013) to build a predictive model and serve it via a REST API. The goal is to predict whether a requester will receive a pizza based on various features like account age, posting history, request content, and timing.

## Recommended Reading Order

1.  `notebooks/preliminary_analysis.ipynb`
2.  `src/pipeline.py`
3.  `notebooks/building_ml_model.ipynb`
4.  `src/train.py`
5.  `src/main.py`
6.  `src/schemas.py`, `src/database.py`, `src/models.py`
7.  `docker-compose.yml` & `Dockerfile`
8.  `tests/`

## 🏗️ Project Structure

```
cybel_test/
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── preliminary_analysis.ipynb  # Data exploration and statistics
│   └── building_ml_model.ipynb     # Model development and training
├── src/                            # Source code
│   ├── main.py                     # FastAPI application
│   ├── pipeline.py                 # ML pipeline implementation
│   ├── database.py                 # Database configuration
│   ├── models.py                   # SQLAlchemy models
│   └── schemas.py                  # Pydantic schemas
├── tests/                          # Test suite
│   └── test_api.py                 # Integration and endpoint tests
├── pizza_request_model.joblib      # The exported model pipeline
├── docker-compose.yml              # Multi-service orchestration
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables
```

## 🔍 Data Analysis Results

### Key Statistics

1.  **Success Rate**: Approximately **24.6%** of pizza requests are fulfilled.
2.  **Top Subreddits**: Most active requesters come from communities like:
    *   `Random_Acts_Of_Pizza`
    *   `funny`
    *   `AskReddit`
    *   `pics`
    *   `WTF`
3.  **Temporal Patterns**:
    *   The average time between requests varies by month.
    *   Only months with 50+ requests were analyzed for statistical significance.
4.  **User Activity**: A rolling 6-month analysis shows seasonal patterns in pizza request activity across different subreddit communities.

The entire analysis can be found in `notebooks/preliminary_analysis.ipynb`.

## 🤖 Machine Learning Approach

### Data Preprocessing

*   **Data Leakage Prevention**: Removed features collected after the request outcome (i.e., `_at_retrieval` columns).
*   **Feature Engineering**:
    *   Text length and content analysis.
    *   Time-based features (hour, day of the week).
    *   User activity ratios (RAOP posts vs. total posts).
    *   Combined text features from title and body.

### Model Pipeline

*   **Preprocessing**:
    *   Numeric features: `StandardScaler`
    *   Categorical features: `OneHotEncoder`
    *   Text features: `TfidfVectorizer` (1000 features, English stopwords)
*   **Algorithm**: Automated model selection using **FLAML** (Fast and Lightweight AutoML).
*   **Evaluation**: **ROC AUC** was used as the primary metric, which is appropriate for imbalanced classification.

### Performance

*   **ROC AUC**: The final model achieved a score of **0.7574**, compared to a baseline of 0.6237.
*   **Cross-validation**: Stratified 5-fold cross-validation was used to handle class imbalance.
*   **Model**: The best-performing algorithm selected by FLAML was a tuned **RandomForestClassifier**.

The entire pipeline and model selection process can be found in `notebooks/building_ml_model.ipynb`, `src/pipeline.py`, and `src/train.py`.

## 🚀 REST API Service

### Features

*   **FastAPI** framework with automatic OpenAPI documentation.
*   **Prediction endpoint**: `POST /predict`
*   **Health check**: `GET /`
*   **Request logging**: All predictions are stored in a database with timestamps.
*   **Input validation**: Pydantic schemas ensure data quality.

### Example Usage

```bash
# Valid prediction request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "request_title": "Broke college student hoping for pizza",
       "request_text_edit_aware": "I have exams tomorrow and would be grateful for a pizza",
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
       "requester_subreddits_at_request": ["funny", "askreddit"],
       "request_id": "t3_example",
       "requester_username": "student123"
     }'
```

**Response:**

```json
{
  "prediction_label": "Pizza Received",
  "prediction_value": 1,
  "probability_of_success": 0.73
}
```

## 🐳 Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services (API + PostgreSQL)
docker-compose up -d --build

# Stop services
docker-compose down
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API (development)
uvicorn src.main:app --reload

# Access API documentation at http://localhost:8000/docs
```

## 🧪 Testing

### Test Coverage

*   Integration tests for database operations and API endpoints.
*   End-to-end tests for full service testing with Docker.
*   Validation tests for input validation and error handling.

### Running Tests

```bash
# Run all tests
pytest tests/ -v
```

## 📊 Model Performance & Evaluation

### Metrics Used

*   **Primary**: ROC AUC (handles class imbalance well).
*   **Secondary**: Precision, Recall, F1-score.
*   **Cross-validation**: Stratified K-fold to ensure representative splits.

### Feature Importance

Top predictive features identified:
1.  Account age at request time
2.  Previous RAOP activity
3.  Request text characteristics
4.  User karma/reputation
5.  Temporal patterns (time of day, day of week)

## 🔧 Configuration

### Environment Variables

The `.env` file is exceptionally included as the data in it is not sensitive.

```bash
# Database (Production)
POSTGRES_USER=pizzadb_user
POSTGRES_PASSWORD=supersecret
POSTGRES_DB=pizzadb
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Testing
TESTING=false  # Set to "true" for test mode
```

## 📈 Future Improvements

I'd like to fine-tune a modern BERT model to try the classification using only the text; the results could be surprising. Some of the other features could potentially be concatenated to the text (e.g., subreddits, time, and day).