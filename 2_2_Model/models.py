import torchvision.models as torchmodels
from torch import nn
import utils as u
# from filterbank import STFT, MelFilter, Log1p, MedFilt
import torch
import os
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor
import torch.nn.functional as F

import pathlib
import soundfile as sf
import torchaudio
import torch

torch.hub.set_dir('/data/woutdecrop/torch/')



import torch
import torch.nn as nn
from transformers import ClapAudioModelWithProjection, AutoProcessor

class CLAPClassifier(nn.Module):
    def __init__(self, model_path, num_classes, sr, device, similarity_matrix, embeddings_path, multi_label=False, layer_size_1=None, layer_size_2=None, layer_size_3=None):
        super(CLAPClassifier, self).__init__()
        
        # Initialize the CLAP model and processor
        self.clap = ClapAudioModelWithProjection.from_pretrained(model_path)
        
        # Define the three linear layers
        self.fc1 = nn.Linear(512, layer_size_1)
        self.fc2 = nn.Linear(layer_size_1, layer_size_2)
        self.fc3 = nn.Linear(layer_size_2, num_classes)
        # self.fc4 = nn.Linear(layer_size_3, num_classes)
        
        # Processor and loss function
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.multi_label = multi_label
        self.device = device
        self.sr = sr
        self.embedding_dir = embeddings_path
        
        # Move similarity matrix to device and initialize custom loss function
        self.loss_func = CustomLossFunction(torch.tensor(similarity_matrix).to(device))
        self.loss_func = CustomLossFunction(similarity_matrix.clone().detach().to(device))

        # Move model parameters to device
        self.to(device)
        
    def forward(self, x, y=None):
        # Move input to the correct device and squeeze if necessary
        out = x.to(self.device).squeeze(1)  # Ensure shape is [batch_size, 512]
        
        # Pass through the linear layers without activation or dropout
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)  # Final layer outputs logits
        # out = self.fc4(out) 
        # Calculate loss if labels are provided
        if y is not None:
            y = y.to(self.device)
            loss = self.loss_func(out, y)
            return loss, out
        
        return out

class CustomLossFunction(nn.Module):
    def __init__(self, similarity_matrix):
        super(CustomLossFunction, self).__init__()
        self.similarity_matrix = similarity_matrix

    def forward(self, outputs, target):
        # print("output", outputs)
        pred_softmax = F.softmax(outputs, dim=-1)
        # print("softmax", pred_softmax.sum(dim=-1))
        
        pred = F.log_softmax(outputs, dim=-1)
        # print('after log softmax', pred)
        # def nll(pred, target, similarity_matrix):
        # Select the similarity row corresponding to the target
        similarity_row = self.similarity_matrix[target]
        # print("simararity", similarity_row)
        # Multiply the input row-wise with the selected similarity row
        tensor = pred * similarity_row
    
        sum_over_columns = torch.sum(tensor, dim=1)
    
        # Take the average over all the rows
        average_over_rows = torch.mean(sum_over_columns)
        
        # print("average", average_over_rows)
        
        # Calculate the negative log likelihood
        return -average_over_rows/sum(sum(self.similarity_matrix))
        
#         return out

# import torch
# import torch.nn as nn
# import os
# from transformers import ClapAudioModelWithProjection, AutoProcessor

# class CLAPClassifier(nn.Module):
#     def __init__(self, model_path, num_classes, sr, device, similarity_matrix, embeddings_path, multi_label=False) -> None:
#         super(CLAPClassifier, self).__init__()
        
#         # Initialize the CLAP model and processor
#         self.clap = ClapAudioModelWithProjection.from_pretrained(model_path)
#         print("fixed clap.")
        
#         # Adding additional layers to increase model complexity
#         self.fc1 = nn.Linear(in_features=512, out_features=512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(in_features=512, out_features=256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(in_features=256, out_features=128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.fc4 = nn.Linear(in_features=128, out_features=64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.fc5 = nn.Linear(in_features=64, out_features=num_classes)
        
#         # Activation function and dropout
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.3)
        
#         self.processor = AutoProcessor.from_pretrained(model_path)
#         self.multi_label = multi_label
#         self.device = device
#         self.sr = sr
        
#         self.embedding_dir = embeddings_path
#         # Move the similarity matrix to the appropriate device
#         self.loss_func = CustomLossFunction(torch.tensor(similarity_matrix).to(device))
        
#         # Move model parameters to the specified device
#         self.to(device)
#         print(self.clap.training)
        
#     def forward(self, x, y=None):
#         # Move input to the correct device
#         out = x.to(self.device)
    
#         # Squeeze to remove the extra dimension if necessary
#         out = out.squeeze(1)  # Now shape should be [batch_size, 512]
#         # print(f"Shape after squeeze: {out.shape}")
    
#         # Pass through layers
#         out = self.dropout(self.relu(self.bn1(self.fc1(out))))
#         # print(f"Shape after bn1: {out.shape}")
        
#         # Continue through additional layers as before
#         out = self.dropout(self.relu(self.bn2(self.fc2(out))))
#         out = self.dropout(self.relu(self.bn3(self.fc3(out))))
#         out = self.dropout(self.relu(self.bn4(self.fc4(out))))
#         out = self.fc5(out)
    
#         # Calculate loss if labels are provided
#         if y is not None:
#             y = y.to(self.device)
#             loss = self.loss_func(out, y)
#             return loss, out
        
#         return out
    



    # def compute_embedding(self, df, desired_fs, max_duration, device, embeddings_path, desc):
    #     embeddings_path_list = []
        
    #     for _, row in df.iterrows():  # Iterate through DataFrame rows
    #         wav_path = row['filename']  # Get the .wav file path from the DataFrame
    #         wav_filename = os.path.basename(wav_path)  # Extract the filename
    #         embedding_filename = os.path.splitext(wav_filename)[0] + '.pt'  # Change extension to .pt
            
    #         # Create the full embedding path
    #         embedding_path = os.path.join(embeddings_path, desc, embedding_filename)
    #         os.makedirs(os.path.dirname(embedding_path), exist_ok=True)  # Ensure directory exists
            
    #         # Load the audio file and adjust the sampling rate if necessary
    #         waveform_info = torchaudio.info(wav_path)
    #         waveform, fs = torchaudio.load(wav_path)
    
    #         if fs != desired_fs:
    #             transform = torchaudio.transforms.Resample(fs, desired_fs)
    #             waveform = transform(waveform)
    #         else:
    #             waveform = waveform
    
    #         # Trim or pad the waveform to the max duration
    #         max_samples = max_duration * desired_fs
    #         waveform = waveform[:, :max_samples]  # Keep up to max_samples
    #         if waveform.shape[1] < max_samples:  # Pad if shorter than max_samples
    #             waveform = F_general.pad(waveform, (0, max_samples - waveform.shape[1]))
    #         waveform = [waveform.cpu().numpy()]
    #         # Process audio input
    #         inputs = self.processor(audios=waveform, return_tensors="pt", sampling_rate=desired_fs, padding=True)
    #         inputs = {key: value.to(self.device) for key, value in inputs.items()}
    
    #         # Generate embedding using CLAP model
    #         out = self.clap(**inputs).audio_embeds.to(self.device)
    
    #         # Save embedding to file
    #         torch.save(out, embedding_path)
    #         print(f"Saved embedding for {wav_filename} at {embedding_path}")
    
    #         # Append the embedding path to the list
    #         embeddings_path_list.append(embedding_path)
    
    #     # Add the embedding paths to the DataFrame
    #     df["embedding"] = embeddings_path_list
    #     return df
    # def compute_embeddings(self, x, y=None, iterator=None, desc=None,wav_path=None):
    #     # Ensure input data is on the correct device
    #     x = [s.cpu().numpy() for s in x]
        
    #     # Create a unique hash for the input data to use as a file identifier
    #     # data_hash = hashlib.sha1(np.concatenate(x).tobytes()).hexdigest()
    #     os.makedirs(os.path.join(self.embedding_dir, desc), exist_ok=True)
    #     wav_path=pathlib.Path(wav_path)
    #     wav_filename = os.path.basename(wav_path)
    #     embedding_filename = os.path.splitext(wav_filename)[0] + '.pt'
    #     embedding_path = os.path.join(self.embedding_dir, desc, embedding_filename)
        
    #     # Check if the embedding is already computed and saved
    #     if os.path.exists(embedding_path):
    #         # Load the saved embedding
    #         out = torch.load(embedding_path, map_location=self.device)
    #         print(f"Loaded saved embedding for input with hash {embedding_path}")
    #     else:
    #         # Process inputs with the processor
    #         inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=self.sr, padding=True)
    #         inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
    #         # Get embeddings from the CLAP model
    #         out = self.clap(**inputs).audio_embeds.to(self.device)
            
    #         # Save the computed embedding
    #         torch.save(out, embedding_path)
    #         # print(f"Saved embedding for input with hash {embedding_path}")
        
    #     # Apply the linear layer
    #     out = self.linear(out)
        
    #     # Calculate loss if labels are provided
    #     if y is not None:
    #         y = y.to(self.device)
    #         loss = self.loss_func(out, y)
    #         return loss, out
    #     return out

        
# def log_softmax(x): return x - x.exp().sum(-1).log().unsqueeze(-1)
# def nll(input, target): return -input[range(target.shape[0]), target].mean()

# pred = log_softmax(x)
# loss = nll(pred, target)
# loss



# pred = pred.to(device)
# target = target.to(device)
# similarity_matrix = similarity_matrix.to(device)
        # return total_loss

class CLAPZeroShotClassifier(nn.Module):
    def __init__(self, model_path, labels, sr, device, multi_label=False) -> None:
        super().__init__()
        print("model!", model_path)
        self.clap = ClapModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.loss_func = nn.CrossEntropyLoss()
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        self.labels = labels
        print("labels", self.labels)
        self.multi_label = multi_label
        self.device = device
        self.sr = sr

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=self.sr, padding=True).to(
          self.device)
        out = self.clap(**inputs).logits_per_audio
        loss = self.loss_func(out, y)
        return loss, out
