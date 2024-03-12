import logging
import sqlite3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the data from the database for a specific year
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
# year = 1  # Choose the desired year
for year in range(2, 13):
    data = load_data_from_db(year)

    data = data[(data['demclas'] != -1) & (data['demclas'] != -9)]

    # Step 2: Prepare the data for training
    # Exclude the target variable and any other unwanted columns
    columns_to_drop = [
        'ad8_1', 'ad8_2', 'ad8_3', 'ad8_4', 'ad8_5', 'ad8_6', 'ad8_7', 'ad8_8', 'ad8_score', 'spid', 'opid',
        f'r{year}demclas', f'hc{year}dementage', 'domain65', f'hc{year}disescn9', f'cg{year}reascano1',
        f'is{year}reasnprx1',
        f'clock65', f'datena65', f'date_prvp', f'clock_scorer', f'wordrecall0_20', f'word65', 'ad8_dem', 'demclas'
    ]
    features = data.columns.difference(columns_to_drop)
    target = 'demclas'

    # Handle missing values, not most_frequent
    imputer = SimpleImputer(strategy='constant', fill_value=-1)
    data[features] = imputer.fit_transform(data[features])

    print("Data after handling missing values:")
    print(data[features].head())

    # Identify numeric and categorical features
    numeric_features = data[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data[features].select_dtypes(include=['object']).columns

    # Create a ColumnTransformer to apply one-hot encoding to categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Apply the preprocessor to the feature data
    X = preprocessor.fit_transform(data[features])

    # Step 3: Split the data into training and testing sets
    #ValueError: Input y contains NaN
    if data[target].isnull().values.any():
        print("There are NaN values in the target variable. Please handle them before proceeding.")
        continue


    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # Step 4: Train a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Step 5: Evaluate the model's performance on the testing set
    accuracy = rf_model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

    # Get the feature importances from the trained model
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    print("All features and their importances:")
    for feature, importance in zip(feature_importances['Feature'], feature_importances['Importance']):
        if importance > 0.5:
            print(f"{feature}: {importance:.4f}")

    # Step 6: Predict demclas for all people in the dataset
    # Get all people in the dataset

    num_people = X_test.shape[0]

    # Create a new DataFrame to store the predictions
    predictions_df = pd.DataFrame(columns=['SPID', 'Actual demclas', 'Predicted demclas'])

    # Iterate over all indices in the test set
    for idx, test_index in enumerate(range(num_people), start=1):
        person = X_test[test_index]
        person_dense = person.toarray().flatten()
        predicted_proba = rf_model.predict_proba([person_dense])

        # Get the predicted class probabilities
        class_probabilities = predicted_proba[0]
        predicted_class_index = np.argmax(class_probabilities)
        predicted_demclas = predicted_class_index + 1  # Directly use the predicted class index

        # Get the actual demclas value for the person from the original data DataFrame
        actual_demclas = y_test.iloc[test_index]

        # Create a new DataFrame with the prediction and concatenate it with predictions_df
        new_prediction_df = pd.DataFrame({
            'SPID': [data.loc[y_test.index[test_index], 'spid']],
            'Actual demclas': [actual_demclas],
            'Predicted demclas': [predicted_demclas]
        })
        predictions_df = pd.concat([predictions_df, new_prediction_df], ignore_index=True)

        print(f"Prediction {idx}:")
        print(f"SPID: {data.loc[y_test.index[test_index], 'spid']}")
        print(f"Actual demclas: {actual_demclas}")
        print(f"Predicted demclas: {predicted_demclas}")
        print(f"Predicted class probabilities: {class_probabilities}")
        # dem score starts at 3,
        # if the predicted class is 3 then the score should be 3
        # if the predicted class is 2 then the score should be 3 - class_probabilities[2]
        # if the predicted class is 1 then the score should be 3 - class_probabilities[2] - (class_probabilities[1] * 2)
        demscore = 0
        for i in range(len(class_probabilities)):
            demscore += (3 - i) * class_probabilities[i]
        print(f"DemScore: {demscore}")
        data.loc[y_test.index[test_index], 'demscore'] = demscore
        # Check if the predicted demclas is different from the actual demclas
        if predicted_demclas != actual_demclas:
            # Plot the predicted probabilities for each class
            classes = range(len(class_probabilities))
            plt.figure(figsize=(8, 4))
            plt.bar(classes, class_probabilities)
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title(f'Predicted Probabilities for SPID {data.loc[y_test.index[test_index], "spid"]}')
            plt.xticks(classes)
            plt.ylim(0, 1)
            plt.tight_layout()

            # Save the graph with a meaningful name and a unique identifier
            #make the spid to no dp
            plt.savefig(f'prediction_graphs/{data.loc[y_test.index[test_index], "spid"]}_prediction_{demscore}.png')

            # Close the plot to avoid memory leaks
            plt.close()

        # save the data to the database
        conn = sqlite3.connect('nhats.db')
        c = conn.cursor()
        c.execute(
            f"UPDATE {table_map[year]['sp']} SET demscore = {demscore} WHERE spid = {data.loc[y_test.index[test_index], 'spid']}")
        c.execute(
            f"UPDATE {table_map[year]['sp']} SET predicted_demclas = {predicted_demclas} WHERE spid = {data.loc[y_test.index[test_index], 'spid']}")
        conn.commit()
        conn.close()

    # Calculate the percentage of different predictions
    different_predictions = predictions_df[predictions_df['Actual demclas'] != predictions_df['Predicted demclas']]
    percentage_different = len(different_predictions) / len(predictions_df) * 100

    # Print the summary
    print("Summary:")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Different predictions: {len(different_predictions)}")
    print(f"Percentage of different predictions: {percentage_different:.2f}%")

    # Add the predicted_demclas column to the original data DataFrame
    data['predicted_demclas'] = np.nan
    data.loc[y_test.index, 'predicted_demclas'] = predictions_df['Predicted demclas']
    # commit the changes to the database
    # Update the database with the predicted_demclas values
