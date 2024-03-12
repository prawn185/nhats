import datetime
import glob
import logging
import os
import sqlite3
from collections import defaultdict
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def preprocess_data(combined_data: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.Series]:
    combined_data = combined_data.dropna(subset=[f'demclas'])
    columns_to_drop = [
        'ad8_1', 'ad8_2', 'ad8_3', 'ad8_4', 'ad8_5', 'ad8_6', 'ad8_7', 'ad8_8', 'ad8_score', 'spid', 'opid',
        f'r{year}demclas', f'hc{year}dementage', 'domain65', f'hc{year}disescn9', f'cg{year}reascano1',
        f'is{year}reasnprx1',
        f'clock65', f'datena65', f'date_prvp', f'clock_scorer', f'wordrecall0_20', f'word65', 'ad8_dem', 'demclas'
    ]

    if year == 5:
        columns_to_drop.append('hc5dementage')
    elif year == 12:
        columns_to_drop.append('hc12dementage')



    X = combined_data.drop(columns_to_drop, axis=1)
    y = combined_data[f'demclas']

    non_numeric_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=non_numeric_columns)

    return X, y


def setup_transformations(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformers = [
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]

    column_transformer = ColumnTransformer(transformers=transformers)

    return column_transformer


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series, year: int, target_var: str) -> Tuple[RandomForestRegressor, float, ColumnTransformer]:
    datetime_str = pd.to_datetime('today').strftime('%Y-%m-%d-%H-%M-%S')
    file_path = os.path.join(os.path.dirname(__file__),
                             f'models//{target_var}/{datetime_str.split("-")[0]}/{datetime_str.split("-")[1]}/{datetime_str.split("-")[2].split("-")[0]}')
    os.makedirs(file_path, exist_ok=True)

    # Split the dataset into training and testing sets
    logging.info(f"Splitting dataset into training and testing sets for year {year}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up the column transformer
    column_transformer = setup_transformations(X)

    # Fit the column transformer on the training data
    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_test)

    logging.info(f"Training Random Forest Model for year {year}")
    n_estimators = 100
    n_jobs = -1  # Use all available cores

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
    rf.fit(X_train_transformed, y_train)

    logging.info(f"Evaluating Model for year {year}")
    y_pred = rf.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Year {year} - MSE: {mse:.5f}")

    dump(rf, f'{file_path}/rf_{target_var}_{year}_mse_{mse:.5f}.joblib')

    return rf, mse, column_transformer


def calculate_feature_importance(rf: RandomForestRegressor, column_transformer: ColumnTransformer, year: int,
                                 mse: float, target_var: str) -> List[Tuple[str, float]]:
    """
    Calculate the feature importance scores for the trained model.

    Args:
        rf (RandomForestRegressor): The trained Random Forest model.
        column_transformer (ColumnTransformer): The column transformer used for feature transformations.
        year (int): The year for which the model was trained.
        mse (float): The mean squared error of the model.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the feature names and their importance scores.
    """
    logging.info(f"Calculating Feature Importance for year {year}")
    importances = rf.feature_importances_
    feature_names = column_transformer.get_feature_names_out()
    feature_importance = sorted(zip(importances, feature_names), reverse=True)

    # Aggregating scores by base feature name
    aggregated_scores = defaultdict(float)
    for importance, name in feature_importance:
        base_feature_name = name.split('__')[-1]  # Adjust based on your naming convention
        aggregated_scores[base_feature_name] += importance

    total_importance = sum(aggregated_scores.values())
    aggregated_scores = {key: (value / total_importance) * 100 for key, value in aggregated_scores.items()}
    sorted_aggregated_scores = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    filtered_scores = [(feature_name, importance) for feature_name, importance in sorted_aggregated_scores if
                       importance >= 1]

    logging.info(f"Feature Importances for year {year}:")
    for feature_name, importance in filtered_scores:
        if importance >= 1:
            logging.info(f"{feature_name}: {importance:.2f}%")
    datetime_str = pd.to_datetime('today').strftime('%Y-%m-%d-%H-%M-%S')
    file_path = os.path.join(os.path.dirname(__file__),
                             f'models//{target_var}/{datetime_str.split("-")[0]}/{datetime_str.split("-")[1]}/{datetime_str.split("-")[2].split("-")[0]}')
    os.makedirs(file_path, exist_ok=True)
    dump(rf, f'{file_path}/rf_{target_var}_{year}_mse_{mse:.5f}.joblib')

    return filtered_scores


def save_feature_importance(filtered_scores: List[Tuple[str, float]], year: int, target_var: str) -> None:
    """
    Save the feature importance scores to a CSV file.

    Args:
        filtered_scores (List[Tuple[str, float]]): A list of tuples containing the feature names and their importance scores.
    year (int): The year for which the feature importance scores were calculated.
    """
    df = pd.DataFrame(filtered_scores, columns=['Feature', 'Importance'])
    df['Importance'] = df['Importance'].apply(lambda x: round(x, 2))
    datetime_str = pd.to_datetime('today').strftime('%Y-%m-%d-%H-%M-%S')
    file_path = os.path.join(os.path.dirname(__file__),
                             f'feature_importances/{target_var}/{datetime_str.split("-")[0]}/{datetime_str.split("-")[1]}/{datetime_str.split("-")[2].split("-")[0]}')
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, f'feature_importance_{target_var}_{datetime_str}year{year}.csv')

    df.to_csv(filename, index=False)
    logging.info(f"Feature importance scores saved to {filename}")


def plot_shap_values(rf: RandomForestRegressor, X_train: np.ndarray, column_transformer: ColumnTransformer,
                     year: int, target_var: str) -> None:
    """
    Calculate and plot the SHAP values for the trained model.

    Args:
        rf (RandomForestRegressor): The trained Random Forest model.
        X_train (np.ndarray): The training feature dataset.
        column_transformer (ColumnTransformer): The column transformer used for feature transformations.
        year (int): The year for which the model was trained.
    """
    logging.info(f"Calculating SHAP values for year {year}")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)
    logging.info("Generating SHAP summary plot")
    datetime_str = pd.to_datetime('today').strftime('%Y-%m-%d-%H-%M-%S')
    file_path = os.path.join(os.path.dirname(__file__),
                             f'shap_plots/{datetime_str.split("-")[0]}/{datetime_str.split("-")[1]}/{datetime_str.split("-")[2].split("-")[0]}')
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, f'shap_summary_plot_{target_var}_{datetime_str}_year_{year}.png')

    logging.info(f"SHAP beeswarm writing")
    shap.summary_plot(shap_values, X_train, feature_names=column_transformer.get_feature_names_out(), show=False)
    plt.savefig(filename)
    plt.close()
    logging.info(f"SHAP summary plot saved to {filename}")

    # Create a new figure for the waterfall plot
    plt.figure()
    shap_values_mean = np.mean(shap_values, axis=0)  # Take the mean of SHAP values across all instances

    # Create an Explanation object
    shap_exp = shap.Explanation(values=shap_values_mean, base_values=explainer.expected_value,
                                feature_names=column_transformer.get_feature_names_out())

    # Create the waterfall plot
    shap.plots.waterfall(shap_exp, max_display=15, show=False)

    # Save the waterfall plot
    waterfall_filename = os.path.join(file_path, f'shap_waterfall_plot_{target_var}_{datetime_str}_year_{year}.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename)
    plt.close()

    logging.info(f"SHAP waterfall plot saved to {waterfall_filename}")


def main(target_var: str, years: int, sample_frac: float = 1.0, shap_enabled: bool = False, year_to_predict: int = 1) -> None:
    # ... (rest of the code remains the same)
    start_time = datetime.datetime.now()
    logging.info(f"Starting script at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for year in range(1, years + 1):
        combined_data = load_data_from_db(year, sample_frac)
        X, y = preprocess_data(combined_data, year)

        rf, mse, column_transformer = train_and_evaluate_model(X, y, year, target_var)
        filtered_scores = calculate_feature_importance(rf, column_transformer, year, mse, target_var)
        save_feature_importance(filtered_scores, year, target_var)
        if shap_enabled:
            plot_shap_values(rf, X, column_transformer, year, target_var)

        logging.info(f"Finished processing data for year {year}")

        # Make a prediction for a random person in the specified year
        if year_to_predict <= years:
            combined_data = load_data_from_db(year_to_predict, sample_frac)
            X, y = preprocess_data(combined_data, year_to_predict)
            column_transformer = setup_transformations(X)
            X_transformed = column_transformer.fit_transform(X)

            # Select a random person from the dataset
            random_index = np.random.randint(0, len(X))
            random_person = X.iloc[random_index]

            # Transform the random person's features using the column transformer
            random_person_transformed = column_transformer.transform(random_person.to_frame().T)

            # Load the trained model for the specified year
            model_file = f"models/{target_var}/{year_to_predict}/rf_{target_var}_{year_to_predict}.joblib"
            rf = joblib.load(model_file)

            # Make a prediction for the random person
            prediction = rf.predict(random_person_transformed)

            logging.info(f"Prediction for a random person in year {year_to_predict}: {random_person_transformed}")
            logging.info(f"Prediction: {{prediction[0]}}")
            logging.info(f"demclas: {{y.iloc[random_index]}}")
        else:
            logging.info(f"No model available for year {year_to_predict}. Please train a model for that year first.")

        # logging.info(f"Prediction for year {year_to_predict}: {prediction}")

    end_time = datetime.datetime.now()
    logging.info(f"Script completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_elapsed_time = (end_time - start_time).total_seconds()
    logging.info(f"Total time taken: {total_elapsed_time / 60:.2f} minutes")





if __name__ == "__main__":
    target_var = "demclas"
    years = 1
    sample_frac = 0.5
    shap_enabled = False
    year_to_predict = 1  # Choose the year for which you want to make a prediction
    main(target_var, years, sample_frac, shap_enabled, year_to_predict)
