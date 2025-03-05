import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Processed Data from Parquet
df = pd.read_parquet("data/processed_data_transposed_with_condition.parquet")

# Load Biomarkers
biomarkers = pd.read_csv("data/biomarkers.csv")["Gene_ID"].tolist()

# Extract Features (X) and Labels (y)
X = df.drop(columns=["Sample_ID", "Condition"])  # Drop non-feature columns
y = df["Condition"]  # Use precomputed Condition column (0=Healthy, 1=AD)

# Ensure Biomarkers Exist in Processed Data
biomarkers = [gene for gene in biomarkers if gene in X.columns]
X = X[biomarkers]  # Keep only selected biomarkers

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
