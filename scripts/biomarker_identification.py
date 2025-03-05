import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import time

start_time = time.time()

# Load processed data
df = pd.read_csv("data/processed_data_fixed.csv", index_col=0)  # Ensure Gene_ID is index

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
    return "Unknown"

# Extract conditions for each sample
sample_ids = df.columns  # GSM IDs
y = pd.Series([get_condition(sid) for sid in sample_ids], name="condition", index=sample_ids)

# Debugging: Check condition distribution
condition_counts = Counter(y.values)
print("Condition Count:", condition_counts)

# Remove "Unknown" samples
valid_indices = y[y != "Unknown"].index
X = df.T  # Transpose so samples are rows
X, y = X.loc[valid_indices], y.loc[valid_indices]  # Keep only valid samples

# Drop genes (columns) with more than 50% missing values
X = X.dropna(thresh=int(0.5 * len(X)), axis=1)
print(f"Filtered genes: {X.shape[1]} remaining")

# Ensure enough samples exist per condition before applying t-test
min_sample_size = 5  # Minimum AD & Healthy samples needed per gene
valid_genes = [
    gene for gene in X.columns
    if (y[y == "AD"].index.isin(X[gene].dropna().index).sum() >= min_sample_size) and
       (y[y == "Healthy"].index.isin(X[gene].dropna().index).sum() >= min_sample_size)
]
print(f"Running t-tests on {len(valid_genes)} valid genes...")

# Perform t-test on valid genes
p_values = {
    gene: ttest_ind(
        X.loc[y == "AD", gene].dropna(),
        X.loc[y == "Healthy", gene].dropna(),
        nan_policy="omit"
    ).pvalue
    for gene in valid_genes
}

# Select significant genes (p < 0.05)
significant_genes = [gene for gene, p in p_values.items() if p < 0.05]
print(f"Significant genes found: {len(significant_genes)}")

# Train Random Forest to find top biomarkers
print(f"Training Random Forest on {len(significant_genes)} significant genes...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X[significant_genes], y)

# Get top features based on importance
feature_importance = pd.Series(rf.feature_importances_, index=significant_genes)
top_genes = feature_importance.nlargest(20).index  # Select top 20 biomarkers

# Save selected biomarkers
biomarkers = list(set(significant_genes) & set(top_genes))
pd.DataFrame(biomarkers, columns=["Gene_ID"]).to_csv("data/biomarkers.csv", index=False)
print(f"Identified {len(biomarkers)} biomarkers. Saving...")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[top_genes])
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(y == "AD").astype(int), cmap="coolwarm", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression Data")
plt.show()

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

print(f"Biomarker identification complete. {len(biomarkers)} biomarkers saved.")