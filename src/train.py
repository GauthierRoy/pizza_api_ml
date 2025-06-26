# src/train.py (The new, improved version with FLAML)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from flaml import AutoML

from .pipeline import build_pipeline

print("Loading raw data...")
raw_df = pd.read_json("data/dataset.json")
raw_df.drop_duplicates(subset=["request_id"], keep="first", inplace=True)

y = raw_df["requester_received_pizza"]
X = raw_df.drop("requester_received_pizza", axis=1)

# Split the data, + a temporary holdout/test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.")


# Get the full pipeline structure, but remove the final classifier step for now.
# This leaves us with just the data cleaning and feature transformation part.
preprocessor_pipeline = build_pipeline()
print("Fitting the preprocessor on the training data...")
X_train_processed = preprocessor_pipeline.fit_transform(X_train, y_train)
X_test_processed = preprocessor_pipeline.transform(X_test)
print(f"Shape of processed training data for FLAML: {X_train_processed.shape}")


print("\nStarting FLAML search on pre-processed data...")
automl = AutoML()
settings = {
    "time_budget": 400,
    "metric": "roc_auc",
    "task": "classification",
    "log_file_name": "src/flaml_run.log",
    "seed": 42,
}
automl.fit(X_train=X_train_processed, y_train=y_train, **settings)
print("FLAML search complete.")

# evaluate the best model found by FLAML
print("\n--- FLAML Results ---")
best_flaml_model = automl.model.estimator
print(f"Best model found: {best_flaml_model.__class__.__name__}")
print(f"Best ROC AUC on internal validation: {1 - automl.best_loss:.4f}")

# evaluate on the holdout test set to confirm performance
test_score = automl.score(X_test_processed, y_test)
print(f"ROC AUC on holdout test set: {test_score:.4f}")


# build
print("\nBuilding the final production pipeline with the best model...")
final_production_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_pipeline),
        ("classifier", best_flaml_model),
    ]
)

print("Fitting the final production pipeline on the full dataset...")
final_production_pipeline.fit(X, y)

# Saving
model_filename = "models/pizza_request_model.joblib"
print(f"Saving final pipeline to {model_filename}...")
joblib.dump(final_production_pipeline, model_filename)
print("Model saved successfully. This file is ready for deployment.")
