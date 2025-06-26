# In main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import joblib

# ... other imports
from schemas import PizzaRequestInput  # Your Pydantic and SQLAlchemy models
from feature_engineering import create_features_for_model
from models import PredictionLog
from database import Base, engine, get_db, IS_TESTING  # Import from our updated database.py

# Always create tables, regardless of testing mode
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pizza Request Success Predictor API",
    description="Predicts whether a Reddit 'Random Acts of Pizza' request will be fulfilled.",
    version="1.0",
)

# 2. Load the trained model pipeline
# This is done once when the application starts
try:
    model = joblib.load("pizza_request_model.joblib")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Model file 'pizza_request_model.joblib' not found.")
    model = None


@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Pizza Prediction API!"}


@app.post("/predict")
def predict_success(request: PizzaRequestInput, db: Session = Depends(get_db)):
    """
    Predicts success and logs the raw request and the outcome.
    """
    # 1. Convert the Pydantic model to a dictionary. This is the "raw" data.
    raw_data_dict = request.model_dump()

    # 2. Create the feature DataFrame for the model
    input_df = create_features_for_model(raw_data_dict)

    # 3. Make prediction
    prediction_array = model.predict(input_df)
    probability_array = model.predict_proba(input_df)
    prediction = int(prediction_array[0])
    probability_of_success = float(probability_array[0][1])
    prediction_label = "Pizza Received" if prediction == 1 else "No Pizza Received"

    # 4. Log to database
    # We store BOTH the raw input and the processed output.
    log_entry = PredictionLog(
        raw_request=raw_data_dict,  # <-- The complete, raw input
        prediction_label=prediction_label,
        prediction_value=prediction,
        probability_of_success=probability_of_success,
    )
    db.add(log_entry)
    db.commit()

    # 5. Return the result to the user
    return {
        "prediction_label": prediction_label,
        "prediction_value": prediction,
        "probability_of_success": probability_of_success,
    }
