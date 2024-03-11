import dask
import pandas as pd
import shap
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy import create_engine
import dask
import logging

# Setup logging for debugging
logging.basicConfig(filename='processing.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dask configuration for improved query planning
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd

# Database connection parameters
db_config = {
    "host": "192.168.0.216",
    "port": "5432",
    "user": "postgres",
    "password": "password",
    "database": "nhats"
}

# Establishing database connection
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(connection_string)

# Fetching table names
table_names_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
table_names_df = pd.read_sql_query(table_names_query, engine)

combined_data = None

# Load and concatenate tables
for table_name in table_names_df['table_name'].tolist():
    try:
        dask_df = dd.read_sql_table(table_name, con=connection_string, index_col='spid', schema='public')
        combined_data = dask_df if combined_data is None else dd.concat([combined_data, dask_df],
                                                                        interleave_partitions=True)
        logging.info(f"Successfully loaded and concatenated {table_name}")
    except Exception as e:
        logging.error(f"Error processing {table_name}: {e}")
        continue

# Preprocessing
try:
    combined_data = combined_data.drop(
        ['ad8_1', 'ad8_2', 'ad8_3', 'ad8_4', 'ad8_5', 'ad8_6', 'ad8_7', 'ad8_8', 'ad8_score'], axis=1)
    logging.info("Dropped unnecessary columns.")
except Exception as e:
    logging.error(f"Error dropping columns: {e}")


def preprocess_df(df):
    # Handle specific columns that may cause IntCastingNaNError
    columns_to_fix = ['r1demclas', 'r2demclas', 'r3demclas', 'r4demclas', 'r5demclas', 'r6demclas', 'r7demclas',
                      'r8demclas']
    for col in columns_to_fix:
        if col in df.columns:
            # Convert to float to handle NaNs gracefully
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Apply the preprocessing function to each partition
meta = combined_data._meta.apply(preprocess_df, axis=0)  # Using apply to infer meta based on the function logic
combined_data = combined_data.map_partitions(preprocess_df, meta=meta)

# Compute combined_data to proceed with non-Dask operations
try:
    X = combined_data.compute()
    logging.info("Successfully computed combined_data.")
except Exception as e:
    logging.error(f"Error computing combined_data: {e}")
    raise

# Assuming 'ad8_dem' is the target variable
y = X.pop('ad8_dem')

# Data type conversion and handling missing values for 'clock_scorer'
X['clock_scorer'] = pd.to_numeric(X['clock_scorer'], errors='coerce')

# Preprocessing with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]),
     X.select_dtypes(include=['int64', 'float64']).columns),
    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))]),
     X.select_dtypes(include=['object']).columns)
])

# Apply transformations
X_transformed = preprocessor.fit_transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Model training
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Feature importance and SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.savefig('shap_summary_plot.png')

# Save the model
dump(rf, 'random_forest_model.joblib')
