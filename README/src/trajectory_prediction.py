import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Retrieve the predicted demclas values for each person across the years
conn = sqlite3.connect('nhats.db')
predicted_demclas_data = pd.read_sql_query("SELECT spid, year, predicted_demclas FROM predicted_demclas", conn)
conn.close()

# Pivot the predicted demclas data
trajectory_data = predicted_demclas_data.pivot(index='spid', columns='year', values='predicted_demclas')

# Filter the trajectory data to include only individuals present for at least 5 years
min_years = 5
trajectory_data_filtered = trajectory_data.dropna(thresh=min_years)

# Analyze the trajectory for each individual
for spid, row in trajectory_data_filtered.iterrows():
    trajectory = row.dropna().tolist()
    print(f"SPID: {spid}")
    print(f"Trajectory: {trajectory}")
    # Perform further analysis or visualization for each trajectory

# Visualize the trajectories using a line plot
plt.figure(figsize=(10, 6))
for spid, row in trajectory_data_filtered.iterrows():
    trajectory = row.dropna()
    plt.plot(trajectory.index, trajectory.values, marker='o', label=f"SPID: {spid}")
plt.xlabel('Year')
plt.ylabel('Predicted Demclas')
plt.title('Demclas Trajectories')
plt.legend()
plt.grid(True)
plt.show()