import logging
import os
import random
import sqlite3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logging.basicConfig(level=logging.INFO)

NUM_YEARS = 12
TRAIN_YEARS = [1, 2]
TEST_YEAR = 12

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

    predicted_demclas_data = pd.DataFrame(columns=['spid', 'year', 'demclas'])

    for year in tqdm(range(1, NUM_YEARS + 1), desc="Loading data"):
        # logging.info(f"Loading predicted_demclas data for year {year}")
        table_name = table_map[year]['sp']
        query = f"SELECT spid, {year} AS year, predicted_demclas FROM {table_name}"
        year_data = pd.read_sql_query(query, conn)
        predicted_demclas_data = pd.concat([predicted_demclas_data, year_data], ignore_index=True, copy=False)
    conn.close()
    return predicted_demclas_data


def create_trajectory(predicted_demclas_data: pd.DataFrame) -> pd.DataFrame:
    trajectory_data = predicted_demclas_data.pivot(index='spid', columns='year', values='predicted_demclas')
    trajectory_data_filtered = trajectory_data.dropna(thresh=NUM_YEARS)  # Keep individuals with data for all years

    if trajectory_data_filtered.empty:
        return pd.DataFrame()  # Return an empty dataframe if no individuals have complete data

    return trajectory_data_filtered


def predict_test_year(trajectory: pd.Series) -> tuple:
    train_years_idx = [year - 1 for year in TRAIN_YEARS]  # Convert years to zero-based indices
    X = trajectory.index.values.reshape(-1, 1)[train_years_idx]  # Use the specified years for training
    y = trajectory.values[train_years_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X, y)
    logging.info(f"-------------------------------------------")
    logging.info(f"Model trained on years: {TRAIN_YEARS}")
    logging.info(f"Model Test Year Prediction: {TEST_YEAR}")
    actual_test_year = trajectory.values[TEST_YEAR - 1]
    predicted_test_year = model.predict([[TEST_YEAR]])[0]

    mse = mean_squared_error([actual_test_year], [predicted_test_year])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error([actual_test_year], [predicted_test_year])
    r2 = r2_score([actual_test_year], [predicted_test_year])

    if actual_test_year == 0:
        accuracy = 0
    else:
        accuracy = round(np.abs(actual_test_year - round(predicted_test_year)))

    reason = f"The model's performance metrics for the {TEST_YEAR}th year prediction:\n" \
             f"RMSE = {rmse:.2f}, MAE = {mae:.2f}, R^2 = {r2:.2f}, Error = {accuracy}"

    return predicted_test_year, actual_test_year, accuracy, reason


def generate_graph(spid, trajectory, predicted_test_year, actual_test_year):
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, NUM_YEARS), trajectory.iloc[:-1], marker='o', linestyle='-', color='blue', label='Previous Demclas')
    plt.plot(NUM_YEARS, actual_test_year, marker='o', linestyle='-', color='turquoise', label='Actual Demclas')
    plt.plot(NUM_YEARS, predicted_test_year, marker='x', linestyle='-', color='red', label='Predicted Demclas')
    plt.plot(NUM_YEARS, round(predicted_test_year), marker='x', linestyle='-', color='orange', label='Predicted Demclas (Rounded)')
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
    plt.ylim( 0.75, 3.2)

    # Create the "prediction_graphs" folder if it doesn't exist
    os.makedirs("prediction_graphs", exist_ok=True)

    # Save the graph as an image file
    plt.savefig(f"prediction_graphs/trajectory_{spid}.png")
    plt.close()

def main():
    predicted_demclas_data = load_predicted_demclas_data()
    trajectory_data = create_trajectory(predicted_demclas_data)

    if trajectory_data.empty:
        logging.warning(f"No individuals found with complete data for all {NUM_YEARS} years.")
        return

    num_individuals = len(trajectory_data)
    logging.info(f"Number of individuals with data for all {NUM_YEARS} years: {num_individuals}")

    accuracies = []
    truth_accuracy = []
    predicted_values = []
    actual_values = []

    for spid, trajectory in tqdm(trajectory_data.iterrows(), total=num_individuals, desc="Processing individuals"):
        # logging.info(f"SPID: {spid}")
        # logging.info(f"Trajectory: {trajectory.tolist()}")

        predicted_test_year, actual_test_year, accuracy, reason = predict_test_year(trajectory)

        # logging.info(f"Predicted {TEST_YEAR}th year: {predicted_test_year:.6f}")
        # logging.info(f"Actual {TEST_YEAR}th year: {actual_test_year:.6f}")
        # logging.info(f"Error Diff: {accuracy}")
        # logging.info(f"Reason: {reason}")
        # logging.info("------------------------------------------------------------------------------------------------")

        accuracies.append(accuracy)
        truth_accuracy.append(actual_test_year)
        predicted_values.append(predicted_test_year)
        actual_values.append(actual_test_year)
        generate_graph(spid, trajectory, predicted_test_year, actual_test_year)

    overall_accuracy = sum(accuracies) / sum(truth_accuracy) * 100
    logging.info(f"Overall Error: {overall_accuracy:.2f}%")
    logging.info(f"Overall Error Diff: {sum(accuracies)}/{sum(truth_accuracy)}")

if __name__ == "__main__":
    main()