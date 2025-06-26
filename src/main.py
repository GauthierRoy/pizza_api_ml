# In main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import joblib
import pandas as pd
from .schemas import PizzaRequestInput
from .models import PredictionLog
from .database import Base, engine, get_db
import os

MODEL_PATH = os.path.join("models", "pizza_request_model.joblib")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pizza Request Success Predictor API",
    description="Predicts whether a Reddit 'Random Acts of Pizza' request will be fulfilled.",
    version="1.0",
)

# Load the trained model pipeline
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    model_pipeline = None
    print("FATAL: pizza_request_model.joblib not found.")


@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Pizza Prediction API!"}


@app.post("/predict")
def predict_success(request: PizzaRequestInput, db: Session = Depends(get_db)):
    """
    Takes raw request data, runs it through the full pipeline,
    predicts success, and logs the request and outcome.
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model is not available.")

    # Convert the pydantic model to a df.
    raw_data_dict = request.model_dump()
    input_df = pd.DataFrame([raw_data_dict])

    # predict using the model pipeline
    try:
        prediction = int(model_pipeline.predict(input_df)[0])
        probability = float(model_pipeline.predict_proba(input_df)[0][1])
        prediction_label = "Pizza Received" if prediction == 1 else "No Pizza Received"
    except Exception as e:
        # can catch issues if the raw data is malformed in a way the pipeline can't handle
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")

    # Log to database
    log_entry = PredictionLog(
        raw_request=raw_data_dict,
        prediction_label=prediction_label,
        prediction_value=prediction,
        probability_of_success=probability,
    )
    db.add(log_entry)
    db.commit()

    return {
        "prediction_label": prediction_label,
        "prediction_value": prediction,
        "probability_of_success": probability,
    }
