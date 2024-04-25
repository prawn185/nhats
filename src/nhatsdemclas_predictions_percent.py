import sqlite3

import pandas as pd
from sqlalchemy import create_engine

# Configure the database connection
SQLITE_DB_FILE = "nhats.db"

# Connect to the SQLite database
conn = sqlite3.connect(SQLITE_DB_FILE)

# Define the number of rounds
num_rounds = 2

# Initialize an empty list to store the DataFrames for each round
dfs = []

# Loop through each round and read the corresponding table from the database
for i in range(1, num_rounds + 1):
    table_name = f'NHATS_Round_{i}_SP_File' if i == 1 else f'NHATS_Round_{i}_SP_file'
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql(query, conn)
    dfs.append(df)

# Merge all DataFrames based on the 'spid' column
merged_df = dfs[0]
for i in range(1, num_rounds):
    merged_df = pd.merge(merged_df, dfs[i], on='spid', how='left')

# Calculate the average values for each round
result = pd.DataFrame()
for i in range(1, num_rounds + 1):
    round_suffix = f'_{i}' if i > 1 else ''
    result[f'avg_demclas_round{i}'] = merged_df[f'demclas'].mean()
    result[f'avg_predicted_demclas_round{i}'] = merged_df[f'predicted_demclas'].mean()
    result[f'avg_diff_round{i}'] = (merged_df[f'predicted_demclas'] - merged_df[f'demclas']).mean()

# Display the result
print(result.to_string(index=False))


# Mild cognitive impairment, how many convert to wrost
# mean time of conversion, like age, round
#from there go one year before can I predict the outcome and with what accuracy

#predict when then go from mild to major

#find those who are stable - like who are mild and stay mild

#find out who sparodic

#find out who has dementia and stay dementia

#then use my algorithm to work out what categrory they're in