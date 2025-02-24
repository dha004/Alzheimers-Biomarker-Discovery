import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def find_data_start(filepath):
    """
    Detects the first row of actual numerical gene expression data.
    Skips metadata and column headers if necessary.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if not line.startswith("!") and "GSM" not in line:  # Ensure we're at numeric data
                return i
    return 0  # Default to 0 if nothing is found (shouldn't happen)

def preprocess_data(input_filepath, output_filepath):
    """
    Preprocesses gene expression data by:
    - Skipping metadata rows
    - Correctly detecting headers
    - Converting numeric columns
    - Removing empty and non-numeric rows
    """
    if not os.path.exists(input_filepath):
        logging.error(f"File not found: {input_filepath}")
        return

    # Identify where actual numerical data starts
    start_line = find_data_start(input_filepath)
    logging.info(f"✅ Data starts at line {start_line + 1} (skipping metadata)")

    try:
        # Read the data, treating the first valid row as the header
        df = pd.read_csv(input_filepath, sep="\t", skiprows=start_line, low_memory=False)

        # Ensure the first row isn't mistakenly treated as data
        if df.iloc[0, 0].startswith("!Sample_geo_accession"):  # Check if metadata still exists
            df = df[1:].reset_index(drop=True)

        # Drop rows where the first column is metadata (non-numeric values)
        df = df[~df.iloc[:, 0].astype(str).str.startswith("!")].reset_index(drop=True)

        # Convert numeric columns while keeping the first column unchanged
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric values to NaN

        # Drop rows/columns that are entirely NaN
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        logging.info("\n🔹 Dataset Info After Processing:")
        logging.info(df.info())
        logging.info(df.head())

        # Save cleaned data
        df.to_csv(output_filepath, index=False)
        logging.info(f"✅ Preprocessed data saved to {output_filepath}")

    except Exception as e:
        logging.error(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    input_filepath = "data/GSE48350_series_matrix.txt"
    output_filepath = "data/processed_data.csv"

    preprocess_data(input_filepath, output_filepath)
