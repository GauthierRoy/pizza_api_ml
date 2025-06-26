# Pizza Request Success Predictor

A machine learning service that predicts whether a Reddit "Random Acts of Pizza" request will be fulfilled, built as part of a take-home challenge for CybelAngel's machine learning engineer position.

## 📋 Project Overview

This project analyzes 4,040 pizza requests from the Reddit community "Random Acts of Pizza" (December 2010 - September 2013) to build a predictive model and serve it via a REST API. The goal is to predict whether a requester will receive a pizza based on various features like account age, posting history, request content, and timing.

## Recommended Reading Order

## 📚 Challenge Question Mapping

### First Part: Building a Model

1. **Get the dataset & Clean the data** 
   - `notebooks/preliminary_analysis.ipynb` - Data loading, cleaning, and quality improvement

2. **Display statistics**
   - `notebooks/preliminary_analysis.ipynb` - All four required statistics:
     * Pizza fulfillment prevalence 
     * Top 10 subreddits analysis
     * 6-month rolling user activity by subreddit
     * Average time between consecutive requests

3. **Preprocess the data for the task**
   - `notebooks/preliminary_analysis.ipynb` - Data exploration and preparation
   - `src/pipeline.py` - Data preprocessing pipeline implementation

4. **Build a machine learning model**
   - `notebooks/building_ml_model.ipynb` - Model development, evaluation, and selection
   - `src/train.py` - Model training script

5. **Export your model**
   - `src/train.py` - Model export functionality
   - `models/pizza_request_model.joblib` - The exported model file

### Second Part: Serving Your Model

6. **Implement a REST service**
   - `src/main.py` - FastAPI application with prediction endpoint
   - `src/schemas.py` - Pydantic schemas for request validation

7. **Containerize the service** 
   - `Dockerfile` - Container configuration for the API service
   - `docker-compose.yml` - Multi-service orchestration (API + PostgreSQL)

8. **Log your predictions**
   - `src/database.py` - Database configuration and connection management
   - `src/models.py` - SQLAlchemy model for prediction logging
   - `src/main.py` - Prediction logging implementation in the API

9. **Test your service**
   - `tests/` - Test suite including:
     * Valid API call tests
     * Invalid API call tests
     * Integration tests for database logging


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
*   **Evaluation**: **F1 Score** was used as the primary metric, which is appropriate for imbalanced classification.

### Performance

*   The selected **Gaussian Naive Bayes** model achieved an **F1-Score of 0.38** and a **ROC AUC of 0.68** on the holdout test set.
*   This result, while modest, was significantly better than other baselines, demonstrating the model's effectiveness at handling the probabilistic nature of the features without overfitting to noise.

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
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "request_id": "t3_w5491",
  "request_text_edit_aware": "I have an exam tomorrow and haven'\''t eaten all day. A pizza would be a lifesaver!",
  "request_title": "Broke college student hoping for a pizza",
  "requester_account_age_in_days_at_request": 365.5,
  "requester_days_since_first_post_on_raop_at_request": 0,
  "requester_number_of_comments_at_request": 25,
  "requester_number_of_comments_in_raop_at_request": 2,
  "requester_number_of_posts_at_request": 10,
  "requester_number_of_posts_on_raop_at_request": 1,
  "requester_number_of_subreddits_at_request": 5,
  "requester_subreddits_at_request": [
    "funny",
    "askreddit",
    "pics"
  ],
  "requester_upvotes_minus_downvotes_at_request": 150,
  "requester_upvotes_plus_downvotes_at_request": 250,
  "requester_username": "studious_student",
  "unix_timestamp_of_request": 1380481858,
  "unix_timestamp_of_request_utc": 1380481858
}
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

You have to comment the `POSTGRES_HOST=db` line in the `.env` file and use `localhost` instead, as the database is not running in a container.

```bash
# 1. Start only the database container
docker compose up db -d

# 2. Install dependencies and run API locally
pip install -r requirements.txt
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
python -m pytest tests -v
```

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
```

## 📈 Future Improvements


The Naive Bayes model provided a robust baseline, but its performance is ultimately limited by the simplicity of the features it was given. The most promising path to a significant improvement lies in extracting deeper, more meaningful signals from the request text.

*   **Deep Text Understanding with a Language Model:** The highest-impact next step would be to fine-tune a modern language model like BERT. Unlike our current word-counting approach, BERT can understand the context, sentiment, and nuance of the request narrative. This could unlock the predictive power hidden in people's stories.

*   **Re-evaluating Complex Models on Better Features:** With powerful text embeddings from a model like BERT, we could then re-evaluate more complex models like LightGBM or a Neural Network. They would finally have a strong enough signal to learn from without overfitting, likely surpassing the simpler Naive Bayes baseline and leading to a state-of-the-art solution.
