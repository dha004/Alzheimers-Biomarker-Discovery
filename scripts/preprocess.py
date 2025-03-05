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
    - Transposes data so samples are rows and genes are columns
    - Adds 'Condition' column (0 for Healthy, 1 for AD)
    - Saves as Parquet for efficient storage
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

        # Transpose data so samples are rows and genes are columns
        df.set_index("Gene_ID", inplace=True)  # Set Gene_ID as index before transposing
        df_transposed = df.T  # Transpose the DataFrame

        # Reset index to ensure sample IDs are in a proper column
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={"index": "Sample_ID"}, inplace=True)  # Rename first column

        # Add 'Condition' column (0 for Healthy, 1 for AD)
        df_transposed["Condition"] = df_transposed["Sample_ID"].apply(lambda x: 0 if x.startswith("GSM300") else (1 if x.startswith("GSM117") else None))

        # Drop rows where Condition is still None (if any invalid sample IDs exist)
        df_transposed.dropna(subset=["Condition"], inplace=True)
        df_transposed["Condition"] = df_transposed["Condition"].astype(int)  # Ensure it's an integer column

        logging.info("\nDataset Info After Processing:")
        logging.info(df_transposed.info())
        logging.info(df_transposed.head())

        # Save as CSV
        df_transposed.to_csv("data/processed_data_transposed_with_condition.csv", index=False)

        # Save transposed data as Parquet (efficient storage format)
        df_transposed.to_parquet(output_filepath, index=False)
        logging.info(f"Preprocessed, transposed, and labeled data saved to {output_filepath} in Parquet format")

    except Exception as e:
        logging.error(f"Error processing data: {e}")

if __name__ == "__main__":
    input_filepath = "data/GSE48350_series_matrix.txt"
    output_filepath = "data/processed_data_transposed_with_condition.parquet"

    preprocess_data(input_filepath, output_filepath)
