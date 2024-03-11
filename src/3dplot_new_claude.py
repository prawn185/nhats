import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed

# Connect to the SQLite database and retrieve the relevant data
conn = sqlite3.connect('nhats.db')
c = conn.cursor()

c.execute("SELECT * FROM NHATS_Round_1_SP_file")
data = c.fetchall()

conn.close()

print("Data retrieved from the database:")
print(data[:5])  # Print the first 5 rows of data

# Preprocess the data
spids = [row[0] for row in data]
features = [row[1:-1] for row in data]
dementia_scores = [row[-1] for row in data]

print("Preprocessed data:")
print("SPIDs:", spids[:5])
print("Features:", features[:5])
print("Dementia Scores:", dementia_scores[:5])

# Convert categorical variables to numeric using one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(features).toarray()

print("Encoded features shape:", encoded_features.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_features, dementia_scores, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Select a random subset of the data for feature selection
subset_indices = np.random.choice(len(X_train), size=1000, replace=False)
X_train_subset = X_train[subset_indices]
y_train_subset = y_train[subset_indices]

# Perform feature selection on the subset
selector = SelectFromModel(rf_classifier, prefit=False)
selector.fit(X_train_subset, y_train_subset)
selected_features = selector.get_support(indices=True)

print("Number of selected features:", len(selected_features))

# Apply the selected features to the entire dataset
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Train the Random Forest classifier with selected features
rf_classifier.fit(X_train_selected, y_train)

# Get the most important features
important_features = [encoder.get_feature_names_out()[i] for i in selected_features]


def predict_dementia(person_data):
    encoded_person_data = encoder.transform([person_data]).toarray()
    person_features = encoded_person_data[:, selected_features]
    predicted_score = rf_classifier.predict(person_features)[0]
    return predicted_score


def plot_results_3d(spids, predicted_scores):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(len(spids))
    ys = np.zeros_like(xs)
    zs = predicted_scores

    ax.scatter(xs, ys, zs, c=zs, cmap='coolwarm', s=50)

    ax.set_xlabel('Person Index')
    ax.set_ylabel('Dummy Axis')
    ax.set_zlabel('Predicted Dementia Score')
    ax.set_title('Dementia Predictions')

    plt.show()


def process_person(person_data):
    predicted_score = predict_dementia(person_data)
    return predicted_score


# Parallelize the processing of multiple persons
predicted_scores = Parallel(n_jobs=-1)(delayed(process_person)(person_data) for person_data in features)

print("Predicted dementia scores:")
print(predicted_scores)

# Plot the predicted dementia scores on a 3D graph
plot_results_3d(spids, predicted_scores)

# Print the most important features
print("Most important features for predicting dementia:")
for feature in important_features:
    print(feature)
