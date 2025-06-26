# src/train.py (The new, improved version with FLAML)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, roc_auc_score, classification_report


from .pipeline import build_pipeline, to_dense

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
print(f"Shape of processed training data: {X_train_processed.shape}")


# Since Naive Bayes needs a dense array, we convert the sparse matrix
X_train_dense = X_train_processed.toarray()
X_test_dense = X_test_processed.toarray()

print("\nTraining and evaluating Naive Bayes model...")
nb_model = GaussianNB()
nb_model.fit(X_train_dense, y_train)

# evaluate on the holdout test set to confirm performance
y_pred = nb_model.predict(X_test_dense)
y_pred_proba = nb_model.predict_proba(X_test_dense)[:, 1]

print("\n--- Naive Bayes Performance on Holdout Test Set ---")
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")


# build

print("\nBuilding the final production pipeline with the best model...")
final_production_pipeline = Pipeline(
    steps=[
        ("preprocessor", build_pipeline()),
        ("to_dense", FunctionTransformer(to_dense)),
        ("classifier", GaussianNB()),
    ]
)

print("Fitting the final production pipeline on the full dataset...")
final_production_pipeline.fit(X, y)

# Saving
model_filename = "models/pizza_request_model.joblib"
print(f"Saving final pipeline to {model_filename}...")
joblib.dump(final_production_pipeline, model_filename)
print("Model saved successfully. This file is ready for deployment.")
