import glob
import os
import random
import sqlite3
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_models(model_directory):
    """
    Load trained models from the specified directory.

    Args:
        model_directory (str): Directory containing the trained models.

    Returns:
        dict: Dictionary mapping year to the corresponding trained model.
    """
    models = {}
    for year in range(1, 13):
        model_path = os.path.join(model_directory, f"rf_demclas_{year}_mse_*.joblib")
        model_files = glob.glob(model_path)
        if model_files:
            model_file = model_files[0]  # Assume only one model file per year
            models[year] = joblib.load(model_file)
    return models


def select_random_persons(year, conn, num_persons):
    """
    Select random persons from the specified year.

    Args:
        year (int): Year to select random persons from.
        conn (sqlite3.Connection): SQLite database connection.
        num_persons (int): Number of random persons to select.

    Returns:
        list: List of tuples containing random persons' data and column names.
    """
    c = conn.cursor()
    table_name = f"NHATS_Round_{year}_SP_File"
    c.execute(f"PRAGMA table_info({table_name})")
    column_names = [row[1] for row in c.fetchall()]
    c.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {num_persons}")
    random_persons = c.fetchall()
    return [(person, column_names) for person in random_persons]


def preprocess_data(person_data, year, column_names, preprocessor):
    """
    Preprocess the person's data using the specified preprocessor.

    Args:
        person_data (tuple): Tuple containing the person's data.
        year (int): Year of the person's data.
        column_names (list): List of column names.
        preprocessor (ColumnTransformer): Preprocessor to transform the data.

    Returns:
        numpy.ndarray: Preprocessed data.
    """
    person_df = pd.DataFrame([person_data], columns=column_names)
    preprocessor.fit(person_df)  # Fit the preprocessor on the person's data
    preprocessed_data = preprocessor.transform(person_df)
    return preprocessed_data


def predict_dementia(models, person_data, year, column_names):
    """
    Predict dementia class for the given person's data using the trained models.

    Args:
        models (dict): Dictionary mapping year to the corresponding trained model.
        person_data (tuple): Tuple containing the person's data.
        year (int): Year of the person's data.
        column_names (list): List of column names.

    Returns:
        dict: Dictionary mapping year to the predicted dementia class.
    """
    preprocessor_path = f"models/demclas/2024/03/09/preprocessor_{year}.joblib"
    preprocessor = joblib.load(preprocessor_path)

    preprocessed_data = preprocess_data(person_data, year, column_names, preprocessor)
    predicted_classes = {}
    for year, model in models.items():
        predicted_class = model.predict(preprocessed_data)
        predicted_classes[year] = predicted_class[0]
    return predicted_classes


def plot_dementia_proximity(predicted_classes_list):
    """
    Plot the dementia proximity for multiple persons.

    Args:
        predicted_classes_list (list): List of dictionaries containing predicted dementia classes for each person.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for predicted_classes in predicted_classes_list:
        years = list(predicted_classes.keys())
        classes = list(predicted_classes.values())
        ax.scatter(years, [0] * len(years), classes)

    ax.set_xlabel("Year")
    ax.set_ylabel("Dummy Axis")
    ax.set_zlabel("Dementia Class")
    ax.set_title("Dementia Proximity for Multiple Persons")

    plt.tight_layout()
    plt.show()


def main():
    model_directory = "models/demclas/2024/03/09"
    db_file = "nhats.db"
    num_persons = 5

    models = load_models(model_directory)

    conn = sqlite3.connect(db_file)
    random_year = 1
    random_persons = select_random_persons(random_year, conn, num_persons)
    conn.close()

    predicted_classes_list = []
    for person_data, column_names in random_persons:
        predicted_classes = predict_dementia(models, person_data, random_year, column_names)
        predicted_classes_list.append(predicted_classes)

    plot_dementia_proximity(predicted_classes_list)


if __name__ == "__main__":
    main()