import logging
import random
import sqlite3
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# Define the table_map as a global variable
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


def load_data_from_db(year: int, sample_frac: float = 1.0) -> pd.DataFrame:

    SQLITE_DB_FILE = "nhats.db"

    # Connect to the SQLite database
    conn = sqlite3.connect(SQLITE_DB_FILE)

    logging.info(f"Processing data for year {year}")

    # Retrieve data from sp and metnonmet, ensuring no duplicated columns
    if year <= 8:
        combined_data = pd.read_sql_query(f"""
            SELECT a.*, b.* FROM {table_map[year]['sp']} a
            LEFT JOIN {table_map[year]['metnonmet']} b ON a.spid = b.spid
            """, conn)
    else:
        combined_data = pd.read_sql_query(f"""
            SELECT * FROM {table_map[year]['sp']}
            """, conn)

    # Remove duplicated columns after the join
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    # Retrieve data from op
    op = pd.read_sql_query(f"""
        SELECT * FROM {table_map[year]['op']}
        """, conn)

    # Merge combined_data with op based on spid
    combined_data = pd.merge(combined_data, op, on="spid", how="left", suffixes=('', '_op'))

    # Retrieve tracker data
    tracker = pd.read_sql_query(f"""
        SELECT * FROM {table_map[year]['tracker']}
        """, conn)

    # Merge combined_data with tracker based on spid
    combined_data = pd.merge(combined_data, tracker, on="spid", how="left", suffixes=('', '_tracker'))

    # Merge additional tables for years 11 and 12
    if year >= 11:
        for table in ['accel_det', 'accel_summ', 'accel_track', 'tab_act']:
            if table in table_map[year]:
                table_data = pd.read_sql_query(f"""
                    SELECT * FROM {table_map[year][table]}
                    """, conn)
                combined_data = pd.merge(combined_data, table_data, on="spid", how="left", suffixes=('', f'_{table}'))

    if year == 12:
        int_inc_imp = pd.read_sql_query(f"""
            SELECT * FROM {table_map[year]['int_inc_imp']}
            """, conn)
        combined_data = pd.merge(combined_data, int_inc_imp, on="spid", how="left", suffixes=('', '_int_inc_imp'))

    conn.close()

    combined_data = combined_data.sample(frac=sample_frac, random_state=42)

    logging.info(f"Data for year {year} read.")

    return combined_data

# Step 1: Load the data from the database for a specific year
year = 12  # Choose the desired year
start_time = time.time()
logging.debug(f"Loading data for year {year}...")
data = load_data_from_db(year)
logging.debug(f"Data loaded in {time.time() - start_time:.2f} seconds")

data = data[(data['demclas'] != -1) & (data['demclas'] != -9)]

# Step 2: Prepare the data for training
# Exclude the target variable and any other unwanted columns
columns_to_drop = ['r1breakoffqt',
    'ad8_1', 'ad8_2', 'ad8_3', 'ad8_4', 'ad8_5', 'ad8_6', 'ad8_7', 'ad8_8', 'ad8_score', 'spid', 'opid',
    f'r{year}demclas', f'hc{year}dementage', 'domain65', f'hc{year}disescn9', f'cg{year}reascano1',
    f'is{year}reasnprx1',
    f'clock65', f'datena65', f'date_prvp', f'clock_scorer', f'wordrecall0_20', f'word65', 'ad8_dem', 'demclas'
]
features = data.columns.difference(columns_to_drop)
target = 'demclas'

# Handle missing values
start_time = time.time()
logging.debug("Handling missing values...")
imputer = SimpleImputer(strategy='constant', fill_value=-1)
data[features] = imputer.fit_transform(data[features])
logging.debug(f"Missing values handled in {time.time() - start_time:.2f} seconds")

# Convert categorical features to numeric using label encoding
start_time = time.time()
logging.debug("Encoding categorical features...")
label_encoders = {}
for feature in features:
    if data[feature].dtype == 'object':
        label_encoders[feature] = LabelEncoder()
        data[feature] = label_encoders[feature].fit_transform(data[feature])
logging.debug(f"Categorical features encoded in {time.time() - start_time:.2f} seconds")

# Identify numeric features
numeric_features = data[features].columns

# Create a ColumnTransformer with only numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features)
    ])

# Apply the preprocessor to the feature data
start_time = time.time()
logging.debug("Applying preprocessor...")
X = preprocessor.fit_transform(data[features])
logging.debug(f"Preprocessor applied in {time.time() - start_time:.2f} seconds")

# Step 3: Train a Random Forest Classifier
data[target] = data[target].fillna(-3)
y = data[target]

start_time = time.time()
logging.debug("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
logging.debug(f"Model trained in {time.time() - start_time:.2f} seconds")

# Step 4: Select a random user and create a risk score
random_user_index = random.randint(0, len(data) - 1)
random_user = data.iloc[random_user_index]
random_user_features = preprocessor.transform([random_user[features]])

# Get the predicted class probabilities for the random user
predicted_proba = rf_model.predict_proba(random_user_features)
class_probabilities = predicted_proba[0]

# Calculate the risk score based on the predicted probabilities
risk_score = class_probabilities[1] - class_probabilities[0]

# Interpret the risk score
if risk_score > 0:
    risk_interpretation = "The user is likely to decline in cognitive ability."
else:
    risk_interpretation = "The user is likely to improve or maintain cognitive ability."

# Print the results
logging.debug(f"Random User: {random_user['spid']}")
logging.debug(f"Actual demclas: {random_user['demclas']}")
logging.debug(f"Predicted class probabilities: {class_probabilities}")
logging.debug(f"Risk Score: {risk_score:.4f}")
logging.debug(f"Risk Interpretation: {risk_interpretation}")

# Step 5: Predict risk scores for all users in the dataset
num_users = len(data)
logging.debug(f"Total users: {num_users}")

risk_scores = []
start_time = time.time()
logging.debug("Predicting risk scores for all users...")
for user_index in tqdm(range(num_users), desc="Predicting risk scores"):
    user = data.iloc[user_index]
    user_features = preprocessor.transform([user[features]])
    predicted_proba = rf_model.predict_proba(user_features)
    class_probabilities = predicted_proba[0]
    risk_score = class_probabilities[1] - class_probabilities[0]
    risk_scores.append(risk_score)
logging.debug(f"Risk scores predicted in {time.time() - start_time:.2f} seconds")

# Print the summary statistics of risk scores
logging.debug(f"Risk Scores - Mean: {np.mean(risk_scores):.4f}")
logging.debug(f"Risk Scores - Median: {np.median(risk_scores):.4f}")
logging.debug(f"Risk Scores - Standard Deviation: {np.std(risk_scores):.4f}")

total_time = time.time() - start_time
logging.debug(f"Total execution time: {total_time:.2f} seconds")