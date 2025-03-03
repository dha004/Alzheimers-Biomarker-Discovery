import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def preprocess_data(input_filepath, output_filepath):
    """
    Preprocess gene expression data:
    - Skips metadata lines
    - Ensures correct headers
    - Separates Gene_ID and ID_REF correctly
    - Drops unnecessary rows
    """
    if not os.path.exists(input_filepath):
        logging.error(f"File not found: {input_filepath}")
        return

    try:
        # Read the file as plain text first
        with open(input_filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Find where the actual data starts
        data_start = next(i for i, line in enumerate(lines) if not line.startswith("!") and "GSM" in line)
        logging.info(f"Data starts at line {data_start + 1}")

        # Read data with corrected headers
        df = pd.read_csv(input_filepath, sep="\t", skiprows=data_start, low_memory=False)

        # Check if first row is duplicated header
        if df.iloc[0, 0] == "ID_REF":
            df = df[1:].reset_index(drop=True)  # Drop duplicate header row

        # Rename first column to "Gene_ID"
        df.rename(columns={df.columns[0]: "Gene_ID"}, inplace=True)

        # Remove rows where Gene_ID is empty or malformed
        df = df[df["Gene_ID"].notna() & df["Gene_ID"].str.match(r"^[a-zA-Z0-9_.-]+$")]

        # Convert numeric columns
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove empty rows and columns
        df.dropna(how="all", axis=0, inplace=True)
        df.dropna(how="all", axis=1, inplace=True)

        logging.info("\nDataset Info After Processing:")
        logging.info(df.info())
        logging.info(df.head())

        # Save cleaned data
        df.to_csv(output_filepath, index=False)
        logging.info(f"Preprocessed data saved to {output_filepath}")

    except Exception as e:
        logging.error(f"Error processing data: {e}")

if __name__ == "__main__":
    input_filepath = "data/GSE48350_series_matrix.txt"
    output_filepath = "data/processed_data_fixed.csv"

    preprocess_data(input_filepath, output_filepath)