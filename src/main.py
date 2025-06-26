# In main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import joblib
import pandas as pd
from schemas import PizzaRequestInput 
from models import PredictionLog
from database import Base, engine, get_db
import os
# Always create tables, regardless of testing mode
MODEL_PATH = os.path.join("models", "pizza_request_model.joblib")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pizza Request Success Predictor API",
    description="Predicts whether a Reddit 'Random Acts of Pizza' request will be fulfilled.",
    version="1.0",
)

# 2. Load the trained model pipeline
try:
    # This path might need to be adjusted based on where you run the app from
    model_pipeline = joblib.load("pizza_request_model.joblib")
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

    # 1. Convert the Pydantic model to a DataFrame.
    # The pipeline expects a DataFrame, even for a single row.
    raw_data_dict = request.model_dump()
    input_df = pd.DataFrame([raw_data_dict])

    # 2. Predict using the model pipeline
    try:
        prediction = int(model_pipeline.predict(input_df)[0])
        probability = float(model_pipeline.predict_proba(input_df)[0][1])
        prediction_label = "Pizza Received" if prediction == 1 else "No Pizza Received"
    except Exception as e:
        # This can catch issues if the raw data is malformed in a way the pipeline can't handle
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


    # 3. Log to database
    log_entry = PredictionLog(
        raw_request=raw_data_dict,
        prediction_label=prediction_label,
        prediction_value=prediction,
        probability_of_success=probability,
    )
    db.add(log_entry)
    db.commit()

    # 4. Return the result
    return {
        "prediction_label": prediction_label,
        "prediction_value": prediction,
        "probability_of_success": probability,
    }
