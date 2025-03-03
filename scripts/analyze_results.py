import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load processed data & selected biomarkers
df = pd.read_csv("data/processed_data_fixed.csv", index_col=0)  # Ensure Gene_ID is index
biomarkers = pd.read_csv("data/biomarkers.csv")["Gene_ID"].tolist()

# Load GSM ID to condition mapping
condition_mapping = {
    "GSM1176": "AD",  # Example pattern
    "GSM300": "Healthy"
}

def get_condition(gsm_id):
    """Determine condition based on GSM ID prefix."""
    for prefix, condition in condition_mapping.items():
        if gsm_id.startswith(prefix):
            return condition
    return "Unknown"  # Fallback

# Extract sample IDs and map to conditions
sample_ids = df.columns  # GSM IDs
conditions = [get_condition(sid) for sid in sample_ids]
y = pd.Series(conditions, name="condition", index=sample_ids)

# Ensure biomarkers exist in processed data
biomarkers = [gene for gene in biomarkers if gene in df.index]
X = df.loc[biomarkers].T  # Transpose so samples are rows

# Align X and y to ensure correct sample matching
X, y = X.align(y, axis=0, join="inner")

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make Predictions
y_pred = rf.predict(X_test)

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# Save Model
joblib.dump(rf, "models/random_forest_model.pkl")

# Print Model Performance
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))