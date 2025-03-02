import pandas as pd

file_path = "data/GSE48350_series_matrix.txt"

# Start reading from the correct line
data_start = 80  # This is the line where ID_REF appears

# Read the file, setting the first row as column headers
df = pd.read_csv(file_path, sep="\t", skiprows=data_start, dtype=str)

# Display the first 10 rows to check the structure
print(df.head(10))
