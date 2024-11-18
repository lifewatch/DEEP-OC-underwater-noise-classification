import pandas as pd
import numpy as np
from renumics import spotlight
# file_path = r'\\fs\SHARED\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\spectrograms\data_with_embeddings.pkl'

file_path = r'\\fs\SHARED\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\spectrograms_updatecolourbar_bigger_update\data_with_embeddings_10sec.pkl'


df = pd.read_pickle(file_path)
# Flatten the arrays in the 'embeddings' column
df['embeddings'] = df['embeddings'].apply(lambda x: list(x))
df['distance'] = df['distance'].apply(lambda x: int(x))
def get_station_and_deployment(file_path):
    station_name = file_path.split('\\')[-1].split("_")[0]
    deployment_number = file_path.split('\\')[-1].split("_")[1]
    activity=file_path.split('\\')[-1].split("_")[-4] 
    return station_name, deployment_number,activity


df[['Station_Name', 'Deployment_Number','activity']] = df['filename'].apply(lambda x: pd.Series(get_station_and_deployment(x)))

df['embeddings_flattened'] = df['embeddings'].apply(lambda x: np.array(x).flatten()


df["distance"]
# df['embedding'] = df['embeddings'].apply(lambda x: list(x.flatten()[0:10]))
# Define the data types for spotlight
dtype = {"embeddings_flattened": spotlight.Embedding}
# dtype = {"spectrogram": spotlight.Image}
# Display the DataFrame using spotlight
spotlight.show(df, dtype=dtype)

