import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure model file exists
model_path = "models/random_forest_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Run train_model.py first.")

# Load Trained Model
rf = joblib.load(model_path)

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

# Make Predictions
y_pred = rf.predict(X)

# Print Classification Report
print("✅ Classification Report:")
print(classification_report(y, y_pred))


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


plot_confusion_matrix(y, y_pred, labels=["Healthy", "AD"])


# Feature Importance Visualization
def plot_top_biomarkers(model, feature_names, top_n=20):
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1][:top_n]  # Get indices of top N features
    top_features = [feature_names[i] for i in sorted_indices]
    top_importances = importances[sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_importances)), top_importances, align="center")
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Biomarkers Identified by Model")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()


plot_top_biomarkers(rf, biomarkers, top_n=20)

print("🔍 Analysis Complete. Check visualizations for details.")
