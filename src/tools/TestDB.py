import psycopg2
import random


def get_random_spids_and_scores(db_params, num_records):
    # db_params should include 'host', 'database', 'user', 'password'
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT spid FROM NHATS_Round_1_SP_File ORDER BY RANDOM() LIMIT %s", (num_records,))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()


# Example usage:
db_params = {
    'host': "192.168.0.216",
    'port': 5432,
    'user': "shaun",
    'password': "riadog",
    'database': "nhats"
}
num_records = 5  # Adjust as needed
random_spids_and_scores = get_random_spids_and_scores(db_params, num_records)
print(random_spids_and_scores)
