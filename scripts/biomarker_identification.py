import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/processed_data.csv")

# Extract condition labels from column names
conditions = ["AD" if "AD" in col else "Healthy" for col in df.columns[1:]]

# Convert to Pandas Series
y = pd.Series(conditions, name="condition", index=df.columns[1:])  # Index matches sample names

# Transpose X so that rows are samples and columns are genes
X = df.set_index("Gene_ID").T  # Transpose and set index to match y

# Align X and y to ensure indices match
X, y = X.align(y, axis=0, join="inner")  # Ensures correct indexing

# 🚨 Drop genes (columns) with more than 50% missing values
X = X.dropna(thresh=int(0.5 * len(X)), axis=1)

# 🚨 Ensure each group has enough samples before applying t-test
min_sample_size = 5  # Minimum AD & Healthy samples needed per gene
valid_genes = [
    gene for gene in X.columns
    if (y[y == "AD"].index.isin(X[gene].dropna().index).sum() >= min_sample_size) and
       (y[y == "Healthy"].index.isin(X[gene].dropna().index).sum() >= min_sample_size)
]

# Perform t-test only on valid genes
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

# Train Random Forest to find top biomarkers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X[significant_genes], y)

# Get top features based on importance
feature_importance = pd.Series(rf.feature_importances_, index=significant_genes)
top_genes = feature_importance.nlargest(20).index  # Select top 20 biomarkers

# Save selected biomarkers
biomarkers = list(set(significant_genes) & set(top_genes))
pd.DataFrame(biomarkers, columns=["Gene_ID"]).to_csv("data/biomarkers.csv", index=False)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[top_genes])
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(y == "AD").astype(int), cmap="coolwarm", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression Data")
plt.show()

print(f"✅ Identified {len(biomarkers)} biomarkers. Saved to data/biomarkers.csv.")

