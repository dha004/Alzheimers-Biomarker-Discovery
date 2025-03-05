import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import time

start_time = time.time()

# Load processed data from Parquet
df = pd.read_parquet("data/processed_data_transposed_with_condition.parquet")

# Extract features (X) and labels (y)
X = df.drop(columns=["Sample_ID", "Condition"])  # Drop non-feature columns
y = df["Condition"]  # Use the precomputed Condition column (0=Healthy, 1=AD)

# Debugging: Check condition distribution
condition_counts = Counter(y)
print("Condition Count:", condition_counts)

# Drop genes (columns) with more than 50% missing values
X = X.dropna(thresh=int(0.5 * len(X)), axis=1)
print(f"Filtered genes: {X.shape[1]} remaining")

# Ensure enough samples exist per condition before applying t-test
min_sample_size = 5  # Minimum AD & Healthy samples needed per gene
valid_genes = [
    gene for gene in X.columns
    if (y[y == 1].index.isin(X[gene].dropna().index).sum() >= min_sample_size) and
       (y[y == 0].index.isin(X[gene].dropna().index).sum() >= min_sample_size)
]
print(f"Running t-tests on {len(valid_genes)} valid genes...")

# Perform t-test on valid genes
p_values = {
    gene: ttest_ind(
        X.loc[y == 1, gene].dropna(),
        X.loc[y == 0, gene].dropna(),
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
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression Data")
plt.show()

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

print(f"Biomarker identification complete. {len(biomarkers)} biomarkers saved.")
