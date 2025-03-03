import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load Trained Model
rf = joblib.load("models/random_forest_model.pkl")

# Load Processed Data & Biomarkers
df = pd.read_csv("data/processed_data_fixed.csv", index_col=0)  # Ensure Gene_ID is index
biomarkers = pd.read_csv("data/biomarkers.csv")["Gene_ID"].tolist()

# Load GSM ID to Condition Mapping
condition_mapping = {
    "GSM1176": "AD",
    "GSM300": "Healthy"
}

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

# Make Predictions
y_pred = rf.predict(X)

# Print Classification Report
print("✅ Classification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=["Healthy", "AD"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "AD"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Visualization
importances = rf.feature_importances_
indices = range(len(importances))
plt.figure(figsize=(12, 6))
plt.barh(indices, importances, align="center")
plt.yticks(indices, biomarkers)
plt.xlabel("Feature Importance")
plt.title("Top Biomarkers Identified by Model")
plt.show()

print("🔍 Analysis Complete. Check visualizations for details.")