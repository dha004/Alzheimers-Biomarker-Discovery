import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load Processed Data
df = pd.read_csv("data/processed_data_transposed_with_condition.csv")

# Extract Features (X) and Labels (y)
X = df.drop(columns=["Condition", "Sample_ID"])
y = df["Condition"]

# Split dataset before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Save test set
joblib.dump((X_test, y_test), "data/test_set.pkl")

# Apply SMOTE only to the training set to avoid data leakage
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Perform biomarker selection using only the training set
def biomarker_selection(X_train, y_train):
    return X_train.columns[:50].tolist()  # Selecting first 50 as an example

selected_biomarkers = biomarker_selection(X_train_resampled, y_train_resampled)

# Reduce training and testing features
X_train_resampled = X_train_resampled[selected_biomarkers]
X_test = X_test[selected_biomarkers]

# Train Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight={0:1, 1:3},
    min_samples_split=4,
    min_samples_leaf=3,
    max_features="sqrt",
    max_depth=12
)
rf.fit(X_train_resampled, y_train_resampled)

# Save Model with Feature Names
joblib.dump({"model": rf, "feature_names": selected_biomarkers}, "models/random_forest_model.pkl")

# Evaluate Model on both Training and Test Data

# Ensure feature alignment before prediction
X_train_resampled = X_train_resampled[selected_biomarkers]
X_test = X_test[selected_biomarkers]

y_train_pred = rf.predict(X_train_resampled)
y_test_pred = rf.predict(X_test)

# Print both classification reports
print("="*50)
print("Classification Report for **Training Data**")
print("="*50)
print(classification_report(y_train_resampled, y_train_pred))

print("\n" + "="*50)
print("Classification Report for **Test Data**")
print("="*50)
print(classification_report(y_test, y_test_pred))

print("\nModel training complete. Saved as models/random_forest_model.pkl")

# Extract and save feature importance
importance_df = pd.DataFrame({"Gene": selected_biomarkers, "Importance": rf.feature_importances_})

# Save the top 20 most important biomarkers
top_20_genes = importance_df.sort_values(by="Importance", ascending=False).head(20)
top_20_genes.to_csv("top_20_biomarkers.csv", index=False)
print("Top 20 Biomarkers saved as top_20_biomarkers.csv")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(top_20_genes["Gene"], top_20_genes["Importance"], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Gene")
plt.title("Top 20 Biomarkers (60/40 Split)")
plt.gca().invert_yaxis()
plt.savefig("top_20_biomarkers.png")
print("Feature importance plot saved as top_20_biomarkers.png")