
from sas7bdat import SAS7BDAT
import os
import tqdm
# Function to convert a single SAS file to CSV
def convert_sas_to_csv(sas_file_path, csv_file_path):
    try:
        with SAS7BDAT(sas_file_path) as file:
            df = file.to_data_frame()
        df.to_csv(csv_file_path, index=False)
        return f"File converted successfully: {csv_file_path}"
    except Exception as e:
        return f"Error converting file {sas_file_path}: {e}"

#src/round_data/sas
sas_directory = 'src/round_data/sas'
csv_directory = 'src/../round_data'


os.makedirs(csv_directory, exist_ok=True)


sas_files = [f for f in os.listdir(sas_directory) if f.endswith('.sas7bdat')]


conversion_results = []
for sas_file in tqdm.tqdm(sas_files):
    sas_file_path = os.path.join(sas_directory, sas_file)
    csv_file_path = os.path.join(csv_directory, sas_file.replace('.sas7bdat', '.csv'))

    rename = csv_file_path.replace('NHATS_Round','')
    result = convert_sas_to_csv(sas_file_path, csv_file_path)
    conversion_results.append(result)



