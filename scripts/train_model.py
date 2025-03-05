import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Processed Data & Biomarkers
df = pd.read_csv("data/processed_data_fixed.csv", index_col=0)  # Ensure Gene_ID is index
biomarkers = pd.read_csv("data/biomarkers.csv")["Gene_ID"].tolist()

# Load GSM ID to Condition Mapping
def load_conditions():
    try:
        conditions = pd.read_csv("data/sample_conditions.csv", index_col=0)["Condition"].to_dict()
        return conditions
    except FileNotFoundError:
        print("Warning: Condition file not found. Using hardcoded mapping.")
        return {"GSM1176": "AD", "GSM300": "Healthy"}

condition_mapping = load_conditions()

def get_condition(gsm_id):
    """Determine condition based on GSM ID prefix."""
    for prefix, condition in condition_mapping.items():
        if gsm_id.startswith(prefix):
            return condition
    return "Unknown"

# Extract Sample IDs & Map Conditions
sample_ids = df.columns  # GSM IDs
y = pd.Series([get_condition(sid) for sid in sample_ids], name="condition", index=sample_ids)

# Ensure Biomarkers Exist in Processed Data
biomarkers = [gene for gene in biomarkers if gene in df.index]
X = df.loc[biomarkers].T  # Transpose so samples are rows

# Align X and y to Ensure Correct Sample Matching
X, y = X.align(y, axis=0, join="inner")

# Sanity Check
print(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts()}")

# Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save Trained Model
joblib.dump(rf, "models/random_forest_model.pkl")
print("Model training complete. Saved as models/random_forest_model.pkl")

# Evaluate Model
y_pred = rf.predict(X_test)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Retraining complete. Run analyze_results.py for further insights.")
