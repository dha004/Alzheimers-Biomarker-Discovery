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

# Load Processed Data
df = pd.read_csv("data/processed_data_transposed_with_condition.csv")

# Load Biomarkers Used in Training
biomarkers = pd.read_csv("data/biomarkers.csv")["Gene_ID"].tolist()

# Ensure Biomarkers Match Model Features
model_features = rf.feature_names_in_
biomarkers = [gene for gene in model_features if gene in df.columns]  # Ensure correct feature set

# Extract Features & Labels
X = df[biomarkers]  # Use the exact features model was trained on
y = df["Condition"]  # Labels (0 = Healthy, 1 = AD)

# Make Predictions
y_pred = rf.predict(X)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Labels match Condition column
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "AD"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(y, y_pred)

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

print("Analysis Complete. Check visualizations for details.")