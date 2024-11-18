import numpy as np
import os
import pandas as pd
d_train_path= r"C:\Users\wout.decrop\environments\Imagine\imagine_environment\CLAP_audio\data/train.txt"
d_valid_path= r"C:\Users\wout.decrop\environments\Imagine\imagine_environment\CLAP_audio\data/val.txt"
d_test_path= r"C:\Users\wout.decrop\environments\Imagine\imagine_environment\CLAP_audio\data/test.txt"
# print("path ", d_train_path)
wavs_folder=r'\\fs\SHARED\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\plots\plots_per_station_4_updated_metadata_extra_filter-window-4'
split = np.genfromtxt(d_train_path, dtype='str', delimiter=' ')
d_train_loc = np.array([os.path.join(wavs_folder, i) for i in split[:, 0]])

split = np.genfromtxt(d_valid_path, dtype='str', delimiter=' ')
d_valid_loc = np.array([os.path.join(wavs_folder, i) for i in split[:, 0]])

split = np.genfromtxt(d_test_path, dtype='str', delimiter=' ')
d_test_loc = np.array([os.path.join(wavs_folder, i) for i in split[:, 0]])

def categorize_speed(speed):
    if speed >= 0 and speed < 8:
        return '0-8'
    elif speed >= 8 and speed < 14:
        return '5-14'
    else:
        return '14+'


def categorize_distance(distance):
    if distance >= 0 and distance < 1000:
        return '0-1 km'
    elif distance >= 1000 and distance < 2000:
        return '1-2 km'
    elif distance >= 2000 and distance < 3000:
        return '2-3 km'
    elif distance >= 3000 and distance < 4000:
        return '3-4 km'
    elif distance >= 4000 and distance < 5000:
        return '4-5 km'
    elif distance >= 5000 and distance < 6000:
        return '5-6 km'
    elif distance >= 6000 and distance < 7000:
        return '6-7 km'
    elif distance >= 7000 and distance < 8000:
        return '7-8 km'
    elif distance >= 8000 and distance < 9000:
        return '8-9 km'
    elif distance >= 9000 and distance <= 10000:
        return '9-10 km'
    else:
        return '10+ km'
    

def process_filenames(d_train):
    # Create DataFrame with filenames
    df = pd.DataFrame({'filename': d_train})
    
    # Extract distance from the filename
    df['distance'] = df['filename'].apply(lambda x: float(x.split('_')[-1].split('.wav')[0]))

    # Extract speed from the filename
    df['speed'] = df['filename'].apply(lambda x: float(x.split('_')[-3].replace('-', '.')))
    df = df[df['speed'] <= 30]
    # Extract activity from the filename
    df['activity'] = df['filename'].apply(lambda x: x.split('_')[-4])

    # Extract ship type from the filename
    df['ship_type'] = df['filename'].apply(lambda x: x.split('_')[-5])

    # Apply the function to create a new column 'distance_category'
    df['distance_category'] = df['distance'].apply(categorize_distance)

    # Apply the function to create a new column 'speed_category'
    df['speed_category'] = df['speed'].apply(categorize_speed)

    # Create a combined_info column
    # df['label'] = df['ship_type'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']
    df['label'] = ["ship"] * len(df['ship_type'])
    # df['activity'] = ["activity"] * len(df['activity'])
    # df['speed_category'] = ["speed_category"] * len(df['speed_category'])
    df['activity'] = ["activity"] * len(df['activity'])
    df['label'] = df['label'] + ' at distance ' + df['distance_category'] + ' with speed ' + df['speed_category'] + ' is ' + df['activity']

    # df['label']= df['distance'].apply(categorize_distance)
    return df

d_train=process_filenames(d_train_loc)
d_valid=process_filenames(d_valid_loc)
d_test=process_filenames(d_test_loc)


from transformers import ClapFeatureExtractor
import torchaudio
import torch
import numpy as np

import torch.nn.functional as F_general

def extract_clap_embeddings(wav_path):
    # Initialize the feature extractor
    feature_extractor = ClapFeatureExtractor()
    desired_fs = 48000
    max_duration = 10
    channel = 0

    waveform_info = torchaudio.info(wav_path)

    # Load audio waveform
    waveform, fs = torchaudio.load(wav_path)

    # Resample if necessary
    if waveform_info.sample_rate != desired_fs:
        transform = torchaudio.transforms.Resample(fs, desired_fs)
        waveform = transform(waveform)

    max_samples = max_duration * desired_fs
    waveform = waveform[channel, :max_samples]

    # Pad waveform if necessary
    if waveform.shape[0] < max_samples:
        waveform = F_general.pad(waveform, (0, max_samples - waveform.shape[0]))

    sr = desired_fs

    # Extract the CLAP embeddings
    mel_spectrogram = feature_extractor(waveform, return_tensors="pt", sampling_rate=sr, padding=True)
    
    return mel_spectrogram

# Example usage:
wav_path = '/srv/CLAP/examples/Grafton_15810_2022-03-15_09-02-23_420-983156_Cargo_underway-using-engine_15-9_2022-03-15-09-09-23_2175.wav'
clap_embeddings = extract_clap_embeddings(wav_path)
print(clap_embeddings)



