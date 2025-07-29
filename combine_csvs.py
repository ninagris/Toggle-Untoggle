import glob
import pandas as pd
import os

# Get all CSV files in the current directory
csv_files = glob.glob("*.csv")

# Load and combine, keeping only the header from the first file
combined_df = pd.concat(
    [pd.read_csv(f) for i, f in enumerate(csv_files)],
    ignore_index=True
)

# Save to a new file
combined_df.to_csv("combined.csv", index=False)