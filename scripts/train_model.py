import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load Processed Data
df = pd.read_csv("data/processed_data_transposed_with_condition.csv")

# Extract Features (X) and Labels (y)
X = df.drop(columns=["Condition", "Sample_ID"])
y = df["Condition"]

# Split dataset before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Apply SMOTE only to the training set to avoid data leakage
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Perform biomarker selection using only the training set
def biomarker_selection(X_train, y_train):
    # Placeholder for actual biomarker selection logic
    return X_train.columns[:50].tolist()  # Selecting first 50 as an example

selected_biomarkers = biomarker_selection(X_train_resampled, y_train_resampled)

# Reduce training and testing features
X_train_resampled = X_train_resampled[selected_biomarkers]
X_test = X_test[selected_biomarkers]

# Train Random Forest Model
rf = RandomForestClassifier(
    n_estimators=200,  # Reduce trees
    random_state=42,
    class_weight={0:1, 1:3},
    min_samples_split=3,  # Require more samples to split
    min_samples_leaf=2,  # Avoid very small leaf nodes
    max_features=15  # Reduce number of features per tree
)
rf.fit(X_train_resampled, y_train_resampled)

# Save Model
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Evaluate Model
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel training complete. Saved as models/random_forest_model.pkl")
