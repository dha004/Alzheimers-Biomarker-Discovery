# Alzheimers-Biomarker-Discovery
Alzheimer’s Disease is a neurodegenerative disorder that affects millions worldwide.
Current diagnosis methods rely on cognitive tests and brain imaging, which are expensive and often detect the disease too late.

This project aims to: 
✔Identify genetic biomarkers that distinguish AD patients from healthy individuals
✔Train a machine learning model that can classify AD vs. Healthy based on gene expression
✔Provide a reproducible pipeline that can be extended for biomarker discovery in other diseases

By leveraging statistical analysis and Random Forest models, this tool helps researchers find potential early indicators of Alzheimer’s Disease.
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
- Python           3.13+
- contourpy        1.3.1
- cycler           0.12.1
- Cython           3.0.12
- fonttools        4.56.0
- git-filter-repo  2.47.0
- imbalanced-learn 0.13.0
- imblearn         0.0
- joblib           1.4.2
- kiwisolver       1.4.8
- matplotlib       3.10.1
- numpy            2.2.3
- packaging        24.2
- pandas           2.2.3
- pillow           11.1.0
- pip              25.0.1
- pyarrow          19.0.1
- pyparsing        3.2.1
- python-dateutil  2.9.0.post0
- pytz             2025.1
- scikit-learn     1.6.1
- scipy            1.15.2
- setuptools       75.8.2
- six              1.17.0
- sklearn-compat   0.1.3
- threadpoolctl    3.5.0
- tzdata           2025.1
- wheel            0.45.1
# References
- Dataset: GSE48350 (NCBI GEO)
- Machine Learning Model: Random Forest Classifier
- Statistical Analysis: T-tests for biomarker selection
