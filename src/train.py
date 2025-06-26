# src/train.py (The new, improved version with FLAML)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from flaml import AutoML

from pipeline import build_pipeline

# --- 1. Load and Prepare Data ---
print("Loading raw data...")
raw_df = pd.read_json("dataset.json")
raw_df.drop_duplicates(subset=["request_id"], keep="first", inplace=True)

y = raw_df["requester_received_pizza"]
X = raw_df.drop("requester_received_pizza", axis=1)

# Split the data into training and a temporary holdout/test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.")


# --- 2. Build a Preprocessing-Only Pipeline ---
# Get the full pipeline structure, but remove the final classifier step for now.
# This leaves us with just the data cleaning and feature transformation part.
preprocessor_pipeline = build_pipeline()
# --- 3. Fit the Preprocessor and Transform Data for FLAML ---
print("Fitting the preprocessor on the training data...")
X_train_processed = preprocessor_pipeline.fit_transform(X_train, y_train)
X_test_processed = preprocessor_pipeline.transform(
    X_test
) 
print(f"Shape of processed training data for FLAML: {X_train_processed.shape}")


# --- 4. Run FLAML to Find the Best Model ---
print("\nStarting FLAML search on pre-processed data...")
automl = AutoML()
settings = {
    "time_budget": 120,
    "metric": "roc_auc",
    "task": "classification",
    "log_file_name": "flaml_run.log",
    "seed": 42,
}
automl.fit(X_train=X_train_processed, y_train=y_train, **settings)
print("FLAML search complete.")


# --- 5. Evaluate the Best Model from FLAML ---
print("\n--- FLAML Results ---")
best_flaml_model = automl.model.estimator
print(f"Best model found: {best_flaml_model.__class__.__name__}")
print(f"Best ROC AUC on internal validation: {1 - automl.best_loss:.4f}")

# Evaluate on the holdout test set to confirm performance
test_score = automl.score(X_test_processed, y_test)
print(f"ROC AUC on holdout test set: {test_score:.4f}")


# --- 6. Build  ---
print("\nBuilding the final production pipeline with the best model...")
final_production_pipeline = Pipeline(
    steps=[
        # Step 1: The entire preprocessing pipeline we defined earlier
        ("preprocessor", preprocessor_pipeline),
        # Step 2: The best classifier found by FLAML
        ("classifier", best_flaml_model),
    ]
)

print("Fitting the final production pipeline on the full dataset...")
final_production_pipeline.fit(X, y)

# --- 7. Save the Final Pipeline ---
model_filename = "pizza_request_model.joblib"
print(f"Saving final pipeline to {model_filename}...")
joblib.dump(final_production_pipeline, model_filename)
print("Model saved successfully. This file is ready for deployment.")
