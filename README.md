# Alzheimers-Biomarker-Discovery
A bioinformatics pipeline for identifying key biomarkers related to Alzheimer's Disease (AD) using gene expression data and machine learning models.
# Features
- Preprocessing: Cleans and normalizes gene expression data
- Biomarker Identification: Selects significant genes using statistical tests & feature importance
- Machine Learning: Trains a Random Forest classifier to distinguish between Healthy vs. AD
- Evaluation: Generates classification reports, confusion matrices, and feature importance plots
# Installation
1. Clone the repository
2. Set up a virtual environment
3. Install dependencies
# Dependencies
- Python 3.13+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- imabalanced-learn
- joblib
# References
- Dataset: GSE48350 (NCBI GEO)
- Machine Learning Model: Random Forest Classifier
- Statistical Analysis: T-tests for biomarker selection
