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

# Load Test Set
test_data_path = "data/test_set.pkl"
if not os.path.exists(test_data_path):
    raise FileNotFoundError("Test set not found. Ensure correct data processing.")

X_test, y_test = joblib.load(test_data_path)

model_data = joblib.load("models/random_forest_model.pkl")
rf = model_data["model"]
feature_names = model_data["feature_names"]

# Ensure test set has the same features as the trained model
X_test = X_test[feature_names]

# Make predictions only on test data
y_pred_test = rf.predict(X_test)

# Save Classification Report to File
report_df = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True)).transpose()
report_df.to_csv("classification_report.csv", index=True)
print("Classification report saved as classification_report.csv")

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Labels match Condition column
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "AD"])

    plt.figure(figsize=(6, 4))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")

# Call function with correct test set arguments
plot_confusion_matrix(y_test, y_pred_test)

# Load and visualize top 20 biomarkers from train_model.py (NOT recomputed)
biomarker_file = "top_20_biomarkers.csv"
if os.path.exists(biomarker_file):
    top_20_genes = pd.read_csv(biomarker_file)

    # Display precomputed feature importance graph (no recomputation)
    plt.figure(figsize=(10, 6))
    plt.barh(top_20_genes["Gene"], top_20_genes["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Gene")
    plt.title("Top 20 Biomarkers (Loaded from train_model.py)")
    plt.gca().invert_yaxis()
    plt.show()  # Display the plot
    print("Feature importance plot displayed.")
else:
    print("Warning: Top 20 biomarker file not found. Run train_model.py first.")