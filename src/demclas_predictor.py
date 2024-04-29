import logging
import os
import random
import sqlite3

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

NUM_YEARS = 1
TRAIN_YEARS = [1]
TEST_YEAR = 2

table_map = {
    1: {
        "sp": "NHATS_Round_1_SP_File",
        "metnonmet": "NHATS_Round_1_MetNonMet",
        "op": "NHATS_Round_1_OP_File_v2",
        "tracker": "NHATS_Round_1_Tracker_File"
    },
    2: {
        "sp": "NHATS_Round_2_SP_File",
        "metnonmet": "NHATS_Round_2_MetNonMet",
        "op": "NHATS_Round_2_OP_File_v2",
        "tracker": "NHATS_Round_2_Tracker_File_v2"
    },
    3: {
        "sp": "NHATS_Round_3_SP_File",
        "metnonmet": "NHATS_Round_3_MetNonMet",
        "op": "NHATS_Round_3_OP_File",
        "tracker": "NHATS_Round_3_Tracker_File_V2"
    },
    4: {
        "sp": "NHATS_Round_4_SP_File",
        "metnonmet": "NHATS_Round_4_MetNonMet",
        "op": "NHATS_Round_4_OP_File",
        "tracker": "NHATS_Round_4_Tracker_File_V2"
    },
    5: {
        "sp": "NHATS_Round_5_SP_File",
        "metnonmet": "NHATS_Round_5_MetNonMet",
        "op": "NHATS_Round_5_OP_File_V2",
        "tracker": "NHATS_Round_5_Tracker_File_V3"
    },
    6: {
        "sp": "NHATS_Round_6_SP_File",
        "metnonmet": "NHATS_Round_6_MetNonMet",
        "op": "NHATS_Round_6_OP_File_V2",
        "tracker": "NHATS_Round_6_Tracker_File_v3"
    },
    7: {
        "sp": "NHATS_Round_7_SP_File",
        "metnonmet": "NHATS_Round_7_MetNonMet",
        "op": "NHATS_Round_7_OP_File",
        "tracker": "NHATS_Round_7_Tracker_File_V2"
    },
    8: {
        "sp": "NHATS_Round_8_SP_File",
        "metnonmet": "NHATS_Round_8_MetNonMet",
        "op": "NHATS_Round_8_OP_File",
        "tracker": "NHATS_Round_8_Tracker_File"
    },
    9: {
        "sp": "NHATS_Round_9_SP_File",
        "op": "NHATS_Round_9_OP_File",
        "tracker": "NHATS_Round_9_Tracker_File"
    },
    10: {
        "sp": "NHATS_Round_10_SP_File",
        "op": "NHATS_Round_10_OP_File",
        "tracker": "NHATS_Round_10_Tracker_File"
    },
    11: {
        "sp": "NHATS_Round_11_SP_File",
        "op": "NHATS_Round_11_OP_File",
        "tracker": "NHATS_Round_11_Tracker_File",
        "accel_det": "NHATS_Round_11_Accel_Det_File",
        "accel_summ": "NHATS_Round_11_Accel_Summ_File",
        "accel_track": "NHATS_Round_11_Accel_Track_File",
        "tab_act": "NHATS_Round_11_Tab_Act_File"
    },
    12: {
        "sp": "NHATS_Round_12_SP_File",
        "op": "NHATS_Round_12_OP_File",
        "tracker": "NHATS_Round_12_Tracker_File",
        "accel_det": "NHATS_Round_12_Accel_Det_File",
        "accel_summ": "NHATS_Round_12_Accel_Summ_File",
        "accel_track": "NHATS_Round_12_Accel_Track_File",
        "tab_act": "NHATS_Round_12_Tab_Act_File",
        "int_inc_imp": "NHATS_Round_12_Int_Inc_Imp_File"
    }
}


def load_predicted_demclas_data() -> pd.DataFrame:
    SQLITE_DB_FILE = "nhats.db"
    conn = sqlite3.connect(SQLITE_DB_FILE)

    predicted_demclas_data = pd.DataFrame()
    for year in tqdm(range(1, NUM_YEARS + 1), desc="Loading data"):
        # logging.info(f"Loading predicted_demclas data for year {year}")
        table_name = table_map[year]['sp']

        # Find the count of each demclas in the current year
        demclas_count_query = f"""
            SELECT demclas, COUNT(*) AS count
            FROM {table_name}
            WHERE demclas >= 1 AND demclas <= 3
            GROUP BY demclas
        """
        demclas_count_df = pd.read_sql_query(demclas_count_query, conn)

        # Find the minimum count among all demclas in the current year
        min_count = demclas_count_df['count'].min()

        # Modify the SQL query to use the minimum count for each demclas
        query = f"""
            SELECT {year} AS year, *
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY demclas ORDER BY rowid) AS rn
                FROM {table_name}
                WHERE demclas >= 1 AND demclas <= 3
            ) t
            WHERE rn <= {min_count}
        """
        year_data = pd.read_sql_query(query, conn)
        predicted_demclas_data = pd.concat([predicted_demclas_data, year_data], ignore_index=True, copy=False)

        # Print the demclas distribution for the current year
        print(f"Demclas distribution for year {year}:")
        print(year_data['demclas'].value_counts())
        print()

        # Print the cumulative demclas distribution
        print("Cumulative demclas distribution:")
        print(predicted_demclas_data['demclas'].value_counts())
        print()

    # Print the final demclas distribution
    print("Final demclas distribution:")
    print(predicted_demclas_data['demclas'].value_counts())

    # preprocess data
    return predicted_demclas_data


def create_trajectory(predicted_demclas_data: pd.DataFrame, min_years: int) -> pd.DataFrame:
    trajectory_data = predicted_demclas_data.pivot(index='spid', columns='year', values='demclas')
    trajectory_data_filtered = trajectory_data.dropna(thresh=min_years)  # Keep individuals with data for at least min_years

    if trajectory_data_filtered.empty:
        return pd.DataFrame()  # Return an empty dataframe if no individuals have the minimum required data

    return trajectory_data_filtered


def predict_test_year(trajectory: pd.Series) -> tuple:
    available_years = trajectory.index[trajectory.notna()]
    train_years_idx = [year - 1 for year in available_years if year != TEST_YEAR]  # Convert years to zero-based indices

    if len(train_years_idx) == 0:
        logging.warning(f"No available training years for SPID: {trajectory.name}")
        return None, None  # Return None if there are no available training years

    X = np.array(train_years_idx).reshape(-1, 1)  # Use the available years for training
    y = trajectory.values[train_years_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X, y)

    conn = sqlite3.connect("nhats.db")
    query = f"""
        SELECT demclas
        FROM {table_map[TEST_YEAR]['sp']}
        WHERE demclas >= 1 AND demclas <= 3
    """

    year_data = pd.read_sql_query(query, conn)
    actual_test_year = year_data['demclas'].values[0] if len(year_data) > 0 else None

    predicted_test_year = model.predict([[TEST_YEAR - 1]])[0] if actual_test_year is not None else None
    return predicted_test_year, actual_test_year


def generate_graph(spid, trajectory, predicted_test_year, actual_test_year):
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, NUM_YEARS), trajectory.iloc[:-1], marker='o', linestyle='-', color='blue',
             label='Previous Demclas')
    plt.plot(NUM_YEARS, actual_test_year, marker='o', linestyle='-', color='turquoise', label='Actual Demclas')
    plt.plot(NUM_YEARS, predicted_test_year, marker='x', linestyle='-', color='red', label='Predicted Demclas')
    plt.plot(NUM_YEARS, round(predicted_test_year), marker='x', linestyle='-', color='orange',
             label='Predicted Demclas (Rounded)')
    plt.xlabel('Year')
    plt.ylabel('Demclas')
    plt.title(f'Demclas Trajectory for SPID: {spid}')
    plt.legend()
    plt.grid(True)

    # Set y-axis ticks and labels from 1 to 3 in increments of 0.5
    y_ticks = np.arange(1, 3.5, 0.5)
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
    # Set y-axis ticks and labels for 1, 2, and 3
    y_ticks = [1, 2, 3]
    y_labels = ['Possible Dementa', 'Probable Dementia', 'No Dementia']
    plt.yticks(y_ticks, y_labels)
    # Set y-axis limits to start from 1
    plt.ylim(0.75, 3.2)

    # Create the "prediction_graphs" folder if it doesn't exist
    os.makedirs("prediction_graphs", exist_ok=True)

    # Save the graph as an image file
    plt.savefig(f"prediction_graphs/trajectory_{spid}.png")
    plt.close()


def main():
    predicted_demclas_data = load_predicted_demclas_data()

    trajectory_data = create_trajectory(predicted_demclas_data, min_years=1)
    #
    if trajectory_data.empty:
        logging.warning(f"No individuals found with complete data for all {NUM_YEARS} years.")
        return

    num_individuals = len(trajectory_data)

    accuracies = []
    truth_accuracy = []
    predicted_values = []
    actual_values = []
    exact = 0
    not_exact = 0
    demclas1 = 0
    demclas2 = 0
    demclas3 = 0

    for spid, trajectory in tqdm(trajectory_data.iterrows(), total=num_individuals, desc="Processing individuals"):
        predicted_test_year, actual_test_year = predict_test_year(trajectory)
        if actual_test_year is None:
            continue  # Skip individuals without the actual test year data

        if actual_test_year == 1:
            demclas1 += 1
        elif actual_test_year == 2:
            demclas2 += 1
        elif actual_test_year == 3:
            demclas3 += 1

        if predicted_test_year == actual_test_year:
            exact += 1
        else:
            not_exact += 1

        accuracy = np.abs(predicted_test_year - actual_test_year)
        accuracies.append(accuracy)
        truth_accuracy.append(actual_test_year)
        predicted_values.append(predicted_test_year)
        actual_values.append(actual_test_year)

    overall_accuracy = (1 - (sum(accuracies) / sum(truth_accuracy))) * 100

    logging.info(f"Exact: {exact}")
    logging.info(f"Not Exact: {not_exact}")
    logging.info(f"Overall Error: {overall_accuracy:.2f}%")
    logging.info(f"Overall Error Diff: {sum(accuracies)}/{sum(truth_accuracy)}")


if __name__ == "__main__":
    main()
