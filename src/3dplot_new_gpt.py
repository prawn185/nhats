import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect('nhats.db')
query = "SELECT * FROM NHATS_Round_1_SP_file"
df = pd.read_sql_query(query, conn)
conn.close()

# Assuming 'spid' is the index and 'demclas' is the target
df.set_index('spid', inplace=True)
target = df['demclas']
features = df.drop('demclas', axis=1)

# Convert non-numeric columns to numeric where applicable, or drop them
# Option 1: Drop non-numeric columns
# features = features.select_dtypes(include=[np.number])

# Option 2: Convert non-numeric values to numeric where possible, and fill in for NaNs
for column in features.columns:
    features[column] = pd.to_numeric(features[column], errors='coerce')  # 'coerce' will set errors to NaN

features = features.fillna(features.mean())

# Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train a model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

num_individuals = 100  # Number of individuals to plot
indices = np.random.choice(X_test.shape[0], num_individuals, replace=False)
predictions = clf.predict(X_test[indices])

# Extracting cg1dclkdraw cg1dclkdraw2 and cg1dclkdraw3
#cg1dclkdraw
cg1dclkdraw =


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with actual demclas values for the Z-axis
ax.scatter(cg1dclkdraw, feature2_values, predictions, c=predictions, cmap='viridis', marker='o')

# Customizing the plot
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Predicted Dementia Class (demclas)')
ax.set_title('3D Plot of Dementia Classification Predictions')

plt.show()



plt.show()
