import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Database connection
conn = sqlite3.connect('nhats.db')

# Rounds and conditions to iterate over
rounds = range(1, 9)  # 8 rounds
conditions = range(1, 9)  # ad8_score conditions

for condition in tqdm(conditions, desc="Conditions"):
    # Initialize a DataFrame to hold all feature importances across rounds
    feature_names_list = []
    feature_importances_list = []


    for round_number in tqdm(rounds, desc=f"Rounds for condition {condition}"):
        table_name = f'NHATS_Round_{round_number}_SP_File'
        df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE ad8_score = {condition}", conn)
        # Preprocessing as per your existing script...
        df = df.dropna(subset=['ad8_score'])
        columns_to_drop = ['ad8_1', 'ad8_2', 'ad8_3', 'ad8_4', 'ad8_5', 'ad8_6', 'ad8_7', 'ad8_8', 'ad8_score', 'spid']
        X = df.drop(columns_to_drop, axis=1)
        y = df['ad8_score']

        # Transformations as per your script...
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        column_transformer = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())]), numeric_columns),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)
        ])
        # Debugging: Print the shape of the DataFrame before imputation
        print("Shape of DataFrame before imputation:", X.shape)

        # Check if the DataFrame is empty
        if not df.empty:

            # Apply transformations, scaling, etc.
            X_transformed = column_transformer.fit_transform(X)

            # More debugging: Print the shape after transformation but before scaling
            print("Shape after transformation, before scaling:", X_transformed.shape)

            # Training the model...
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Feature Importances...
            feature_importances = rf.feature_importances_
            # Get feature names
            feature_names = np.array(column_transformer.get_feature_names_out())
            # Sort the feature importances in descending order and take the top 15
            indices = np.argsort(feature_importances)[::-1][:15]
            # Slice out the top 15 feature names and their corresponding importances
            top_feature_names = feature_names[indices]
            top_importances = feature_importances[indices]
            # Normalize the top importances to sum to 1 and convert to percentages
            top_importances_normalized = (top_importances / top_importances.sum()) * 100

            # Append the top feature names and their importances to the lists
            feature_names_list.append(top_feature_names)
            feature_importances_list.append(top_importances_normalized)

        else:
            print(f"DataFrame for round {round_number} is empty")
            print(f"SQL: SELECT * FROM {table_name} WHERE ad8_score = {condition}")

        # Convert lists to DataFrame
        all_feature_importances = pd.DataFrame({
            f'Feature_Round_{i + 1}': names for i, names in enumerate(feature_names_list)
        })
        all_feature_importances_percentage = pd.DataFrame({
            f'Importance_Round_{i + 1} (%)': importances for i, importances in enumerate(feature_importances_list)
        })

        # Combine feature names and importances into one DataFrame
        combined_feature_importances = pd.concat([all_feature_importances, all_feature_importances_percentage], axis=1)

        # Write the DataFrame to a CSV file
        print("Writing to CSV...")
        combined_feature_importances.to_csv(f'feature_importances_ad8_score_{condition}.ft_importances', index=False)
        print("Done!")
