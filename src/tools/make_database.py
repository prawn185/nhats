import os
import sqlite3
import pandas as pd
from tqdm import tqdm

# SQLite database file path
SQLITE_DB_FILE = "nhats.db"

# Connect to the SQLite database
conn = sqlite3.connect(SQLITE_DB_FILE)

# Define the paths to your CSV files
# Loop through the CSV files and create a dictionary of CSV files
csv_paths = {}
for root, dirs, files in os.walk("../round_data"):
    for file in files:
        if file.endswith(".csv"):
            csv_paths[file.replace('.csv', '')] = os.path.join(root, file)

# Iterate over the CSV files and import them into SQLite one by one
for table_name, csv_path in tqdm(csv_paths.items(), desc="Processing tables", unit="table"):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path, low_memory=False)

    # Clean column names: replace periods and spaces with underscores
    df.columns = [c.replace('.', '_').replace(' ', '_') for c in df.columns]

    # Write the DataFrame to the SQLite database
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    tqdm.write(f"Imported data from CSV to SQLite table: {table_name}")

# Close the SQLite connection
conn.close()

print("Data import from CSV to SQLite completed successfully.")