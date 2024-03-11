import sqlite3

conn = sqlite3.connect('nhats.db')  # Adjust the database name as needed
cur = conn.cursor()

# Initialize counters
total_rows = 0
total_columns = 0

# Get all table names
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

for table in tables:
    table_name = table[0]

    # Count rows in each table
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cur.fetchone()[0]
    total_rows += row_count  # Accumulate total row count
    print(f"{table_name}")

    # Count columns in each table
    cur.execute(f"PRAGMA table_info({table_name});")
    columns = cur.fetchall()
    column_count = len(columns)
    total_columns += column_count  # Accumulate total column count
    # print(f"Table {table_name}")


print(f"The database has a total of {total_rows} rows and {total_columns} columns.")
print(f"Total cells in the database: {total_rows * total_columns}")


#
# #Calculate the total number people who have an ad_score of more than 0 across all years
# total_ad8 = 0
# for table in tables:
#     table_name = table[0]
#     cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE ad8_score > 0;")
#     ad8_count = cur.fetchone()[0]
#     total_ad8 += ad8_count
#     print(f"Table {table_name} has {ad8_count} people with AD8 score > 0.")
# print(f"Total people with AD8 score > 0 across all tables: {total_ad8}")
#
