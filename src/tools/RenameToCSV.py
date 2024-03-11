import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ft_importance"):
            old_path = os.path.join(directory, filename)
            new_filename = filename[:-14] + ".csv"  # Remove ".ft_importance" and add ".csv"
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage
directory_path = "../../round_data"
rename_files(directory_path)