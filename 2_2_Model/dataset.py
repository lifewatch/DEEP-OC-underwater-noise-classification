import os
import pathlib
import datetime
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import csv
import utils as u
import models
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import re

import ray  # Ensure Ray is imported
# from ray import tune

class CLAP_Vessel_Distance:
    def __init__(self, config, weights=None):
        self.config = config
        self.weights = weights or {
            'distance_weight': 1,
            'speed_weight': 0,
            'activity_weight': 0,
            'vessel_type_weight': 0
        }
        self._setup_params(config)

        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    def _create_folder(self):
        self.weights_str = '_'.join([f"{u.float_to_string(value)}" for _, value in self.weights.items()])+ "_" + str(round(self.a,3)) + "_" + str(round(self.b,3)) 

        self.result_dir = f"/srv/CLAP/3_1_Results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{self.weights_str}"
        self.model_folder = os.path.join(self.result_dir, "model")
        os.makedirs(self.model_folder, exist_ok=True)

    def _setup_params(self, config):
        """Initialize parameters from the config."""
        self.duration = config['duration']
        self.overlap = config['overlap']
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.wavs_folder = pathlib.Path(config['wavs_folder'])
        self.d_train_path = config['d_train_path']
        self.d_valid_path = config['d_valid_path']
        self.d_test_path = config['d_test_path']
        self.model_path='davidrrobinson/BioLingual'
        self.embeddings_path='/srv/CLAP/embeddings'
        # self.model_path='/srv/CLAP/3_1_Results/2024-11-07_12-19_1_0_0_0_-3.0_0.4/model'
    def _initialize_device(self):
        """Check for CUDA availability and set the device accordingly."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Selected CUDA device: {torch.cuda.get_device_name(device)}")
        else:
            print("CUDA is not available. Using CPU.")
            device = torch.device('cpu')
        return device
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file_path = os.path.join(self.result_dir, 'logs.log')
        self.logging=logging
        self.logging.basicConfig(filename=log_file_path,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # logging.info(f"Configuration: lr = {self.config.get('lr')}, batch_size = {self.batch_size}, epochs = {self.epochs}")
        self.logging.info(f"Results directory: {self.result_dir}")

    def _prepare_dataloaders(self, batch_size, stop_shuffle):
        """Prepare dataloaders for training, validation, and test sets."""
        d_train_loc = self._load_dataset_paths(self.d_train_path)
        d_valid_loc = self._load_dataset_paths(self.d_valid_path)
        d_test_loc = self._load_dataset_paths(self.d_test_path)

        d_train, d_valid, d_test = self._process_datasets(d_train_loc, d_valid_loc, d_test_loc)

        model, similarity_matrices = self._initialize_model(
            a=self.a,
            b=self.b
        )
        self.model=model
        self.similarity_matrices=similarity_matrices

        # Define paths
        train_path = os.path.join(self.embeddings_path, "train")
        valid_path = os.path.join(self.embeddings_path, "Validation")
        test_path = os.path.join(self.embeddings_path, "Test")
        # Create directories if they don't exist
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        # Helper function to check if a directory is empty
        def is_empty(directory):
            return not os.listdir(directory)  # Returns True if directory is empty
        
        # Check if the directories are empty and act accordingly
        if is_empty(train_path) and is_empty(valid_path) and is_empty(test_path):
            print('is empty')
            # Compute and save embeddings if directories are empty
            d_train = model.compute_embedding(d_train, desired_fs=self.desired_fs, max_duration=self.duration, device=self.device, embeddings_path=self.embeddings_path, desc="train")
            d_valid = model.compute_embedding(d_valid, desired_fs=self.desired_fs, max_duration=self.duration, device=self.device, embeddings_path=self.embeddings_path, desc="Validation")
            d_test = model.compute_embedding(d_test, desired_fs=self.desired_fs, max_duration=self.duration, device=self.device, embeddings_path=self.embeddings_path, desc="Test")
        else:
            print('created already')
            # If directories are not empty, add embedding paths to DataFrames
            def add_embedding_column(df, desc):
                # Update each DataFrame with the computed embedding paths
                df["embedding"] = df["filename"].apply(lambda x: os.path.join(self.embeddings_path, desc, os.path.splitext(os.path.basename(x))[0] + '.pt'))
                return df
        
            # Apply the function to each dataset
            d_train = add_embedding_column(d_train, "train")
            d_valid = add_embedding_column(d_valid, "Validation")
            d_test = add_embedding_column(d_test, "Test")


        dataloader_train = self._create_dataloader(d_train,"train" ,batch_size, stop_shuffle)
        dataloader_val = self._create_dataloader( d_valid,"Validation",batch_size, stop_shuffle)
        dataloader_test = self._create_dataloader( d_test,"Test",batch_size, True)

        return dataloader_train, dataloader_val, dataloader_test

    def _load_dataset_paths(self, file_path):
        return np.array([os.path.join(self.wavs_folder, i) for i in np.genfromtxt(file_path, dtype='str', delimiter=' ')[:, 0]])

    def _process_datasets(self, d_train_loc, d_valid_loc, d_test_loc):
        d_train = u.process_filenames(d_train_loc)#.iloc[0:50]
        d_valid = u.process_filenames(d_valid_loc)#.iloc[0:50]
        d_test = u.process_filenames(d_test_loc)#.iloc[0:50]

        valid_labels = self._filter_labels(d_train)
        d_train = d_train[d_train['label'].isin(valid_labels)]
        d_valid = d_valid[d_valid['label'].isin(valid_labels)]
        d_test = d_test[d_test['label'].isin(valid_labels)]

        # Initialize self.ids
        descriptions = d_train['label'].unique()
        sorted_descriptions = sorted(descriptions, key=lambda desc: u._sort_description(desc))
        self.ids = {lbl: i for i, lbl in enumerate(sorted_descriptions)}
        self.d_test=d_test
        return d_train, d_valid, d_test

    def _filter_labels(self, d_train):
        """Filter out infrequent labels with fewer than 5 occurrences."""
        label_counts = d_train['label'].value_counts()
        return label_counts[label_counts >= 5].index

    def _create_dataloader(self, df,desc, batch_size, stop_shuffle):
        return torch.utils.data.DataLoader(
            dataset=u.DatasetLoadEmbeddings(df=df,ids=self.ids,device=self.device),
            batch_size=batch_size,
            shuffle=not stop_shuffle
        )

    def _save_csv(self, predicted_list, true_values_list, predicted_list_numbers, true_values_list_numbers, weights):
        csv_filename = 'predicted_true_values_{}_{}_{}.csv'.format(self.timestamp, self.weights_str,"results")
        csv_file_path = os.path.join(self.result_dir, csv_filename)
        
        # Create a DataFrame with the required columns
        df = pd.DataFrame({
            'predicted': predicted_list,
            'true': true_values_list,
            'predicted_numbers': predicted_list_numbers,
            'true_numbers': true_values_list_numbers,
        })
        
        # Extract timestamps from filenames using regex

    
        # Reset index for both the DataFrame and the d_test for concatenation
        self.d_test.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Concatenate along columns axis (axis=1) if needed
        df = pd.concat([self.d_test, df], axis=1)

        def extract_timestamps(filename):
            # Regular expression to match timestamps in the format YYYY-MM-DD_HH-MM-SS
            timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
            timestamps = re.findall(timestamp_pattern, filename)
            return timestamps[0] if timestamps else None
        
        # Apply the function to extract timestamps and create a new column in the DataFrame
        df['timestamp'] = df['filename'].apply(extract_timestamps)
        
        # Optionally format the timestamp by replacing underscores with dashes
        df['timestamp'] = df['timestamp'].str.replace('_', '-')
        df = df.drop(columns=['label'])
        # Save the combined DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        self.logging.info(f"CSV file saved successfully at {csv_file_path}")


    def _save_figure(self, actual_values, predicted_values):
        # Compute confusion matrix
        labels=sorted(set(actual_values) | set(predicted_values))
        labels.sort()
        # print(labels)

        # print(labels)
        km_labels = u.convert_labels_to_km(labels)
        actual_values_cat = u.convert_labels_to_km(actual_values)
        predicted_values_cat = u.convert_labels_to_km(predicted_values)
        print(km_labels)
        # Create confusion matrix using the original numeric labels
        cm = confusion_matrix(actual_values_cat, predicted_values_cat, labels=km_labels)
        # print(cm)
        # Calculate the MSE
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = round(np.sqrt(mse),3)
        if rmse>3:
            # shutil.rmtree(self.result_dir, ignore_errors=True)
            return
        print(f"Root Mean Squared Error (RMSE): {round(rmse,3)} km for a:{self.a} and b:{self.b}")

        CM_filename = 'CM_{}_{}_{}_{}_{}_bs_{}.png'.format(self.timestamp, self.weights_str,round(rmse,3),self.batch_size)
        CM_file_path_all = os.path.join(self.result_dir, CM_filename)
        
        u.plot_confusion_matrix(cm, classes=km_labels,title=f"CM with RMSE: {rmse} for a:{round(self.a,3)} and b:{round(self.b,3)} and batch_size: {self.batch_size}", save_path=CM_file_path_all)
        
        CM_file_path_all = os.path.join("/srv/CLAP/3_3_figures/confusion_matrices/", CM_filename )
        u.plot_confusion_matrix(cm, classes=labels,title=f"CM with RMSE: {rmse}", save_path=CM_file_path_all)

    def eval_pytorch_model(self, model, dataloader, similarity_matrices, desc):
        """Evaluate the model and return loss, metrics, and MSE."""
        model.eval()
        total_loss = 0.0
        steps = 0
        metrics = {key: [] for key in similarity_matrices.keys()}
        MSE_list = []

        true_values_list = []
        predicted_list = []


        true_values_list_numbers = []
        predicted_list_numbers = []
        
        progress_bar = tqdm(dataloader, desc=desc, leave=True)
        iterator=0
        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)

                loss, logits = model(x, y)
                logits=logits.squeeze(1)
                iterator+=1
                total_loss += loss.cpu().item()
                steps += 1

                for key, sim_matrix in similarity_matrices.items():
                    metrics[key] = u.metrics_calculator(sim_matrix, logits, metrics[key], y)

                predicted_number = logits.argmax(dim=1).cpu().numpy()
                true_values_number = y.cpu().numpy()

                true_values_list_numbers.extend(true_values_number)
                predicted_list_numbers.extend(predicted_number)


                predicted = u.max_finder(logits, self.ids)
                true_values = [list(self.ids.keys())[list(self.ids.values()).index(idx)] for idx in y.tolist()]
                predicted_list.extend(predicted)
                true_values_list.extend(true_values)

                
                MSE_batch = mean_squared_error(true_values_number, predicted_number)
                MSE_list.append(MSE_batch)

                rounded_metrics = {key: round(np.mean(metrics[key]), 4) for key in metrics}
                rounded_MSE = round(MSE_batch, 4)

                progress_bar.set_postfix({
                    'Loss': f'{total_loss/steps:.4f}',
                    'MSE': f'{rounded_MSE:.4f}',
                    **{key: f'{val:.4f}' for key, val in rounded_metrics.items()}
                })

        metrics_avg = {key: round(np.mean(metrics[key]), 4) for key in metrics.keys()}
        MSE_avg = round(np.mean(MSE_list), 4)

        self.logging.info(f"{desc} | Loss: {total_loss/steps:.4f} | Metrics: {metrics_avg} | MSE: {MSE_avg}")
        print(f"{desc} | Loss: {total_loss/steps:.4f} | Metrics: {metrics_avg} | MSE: {MSE_avg}")
        if desc == "Test":
            self._save_csv(predicted_list, true_values_list,predicted_list_numbers,true_values_list_numbers, self.weights)
            self._save_figure(true_values_list_numbers, predicted_list_numbers)
        return total_loss / steps, metrics_avg, MSE_avg 

    def _initialize_model(self, a, b):
        """Initialize and return the model with similarity matrices."""
        # Check which weights are non-zero
        non_zero_weights = {k: v for k, v in self.weights.items() if v != 0}
        
        # Initialize the similarity matrices
        similarity_matrices = {}
        
        if len(non_zero_weights) > 1:
            # If more than one weight is non-zero, initialize the Metric matrix and specific matrices
            similarity_matrices['Metric'] = u.similarity(self.ids, self.device, a, b, **self.weights)
            
            if 'distance_weight' in non_zero_weights:
                similarity_matrices['distance'] = u.similarity(self.ids, self.device, a, b, distance_weight=1)
            if 'speed_weight' in non_zero_weights:
                similarity_matrices['speed'] = u.similarity(self.ids, self.device, a, b, speed_weight=1)
            if 'activity_weight' in non_zero_weights:
                similarity_matrices['activity'] = u.similarity(self.ids, self.device, a, b, activity_weight=1)
            if 'vessel_type_weight' in non_zero_weights:
                similarity_matrices['type'] = u.similarity(self.ids, self.device, a, b, vessel_type_weight=1)
        else:
            # If only one weight is non-zero, initialize only the Metric matrix
            similarity_matrices['Metric'] = u.similarity(self.ids, self.device, a, b, **self.weights)
        
        # Initialize the model with the Metric similarity matrix
        model = models.CLAPClassifier(
            self.model_path,
            num_classes=len(self.ids),
            sr=self.desired_fs,
            device=self.device,
            similarity_matrix=similarity_matrices['Metric'],
            embeddings_path=self.embeddings_path,
            multi_label=False,
            layer_size_1=self.layer_size_1, 
            layer_size_2=self.layer_size_2,
            layer_size_3=self.layer_size_3
        )
        model = model.to(self.device)
        print(model.clap.training)
        return model, similarity_matrices


    def _train_model(self, model, dataloader_train, dataloader_val, optimizer, scheduler, epochs, patience, similarity_matrices):
        """Train the model with early stopping, logging, and additional metrics."""
        best_metric = float('-inf')
        early_stopping_counter = 0
        # epochs=1

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for epoch in range(epochs):
            # Freeze CLAP parameters
            for param in model.clap.parameters():
                param.requires_grad = False
            

            total_loss = 0.0
            steps = 0
            metrics = {key: [] for key in similarity_matrices.keys()}
            MSE_list = []

            progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
            iterator=0
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss, logits = model(x, y)
                iterator+=1
                loss.backward()
                optimizer.step()

                total_loss += loss.cpu().item()
                steps += 1
                # logits=logits[0]
                logits=logits.squeeze(1)
                
                # Update metrics and MSE
                for key, sim_matrix in similarity_matrices.items():
                    metrics[key] = u.metrics_calculator(sim_matrix, logits, metrics[key], y)

                predicted = logits.argmax(dim=1).cpu().numpy()
                true_values = y.cpu().numpy()
                MSE_batch = mean_squared_error(true_values, predicted)
                MSE_list.append(MSE_batch)

                rounded_metrics = {key: round(np.mean(metrics[key]), 4) for key in metrics}
                rounded_MSE = round(MSE_batch, 4)

                progress_bar.set_postfix({
                    'Loss': f'{total_loss/steps:.4f}',
                    'MSE': f'{rounded_MSE:.4f}',
                    **{key: f'{val:.4f}' for key, val in rounded_metrics.items()}
                })

            avg_loss, val_metrics, val_mse = self.eval_pytorch_model(model, dataloader_val, similarity_matrices, "Validation")

            # Report the validation loss to Ray Tune
            # if hasattr(ray, 'tune'):  # Check if Ray Tune is available
                # tune.report(val_loss=avg_loss)
            ray.train.report(dict(val_mse=val_mse))
            
            scheduler.step(avg_loss)
            # Log the current learning rate after the scheduler step
            for param_group in optimizer.param_groups:
                self.logging.info(f"Current learning rate: {param_group['lr']}")
    
            if val_metrics['Metric'] > best_metric:
                best_metric = val_metrics['Metric']
                # self._save_model(model)
                self.best_model=model
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                self.logging.info("Early stopping triggered.")
                break

            # Log metrics and MSE for the epoch
            metrics_avg = {key: round(np.mean(metrics[key]), 4) for key in metrics.keys()}
            MSE_avg = round(np.mean(MSE_list), 4)
            logging.info(f"Epoch {epoch + 1}/{epochs} | Train metrics: {' | '.join([f'{key}: {metrics_avg[key]}' for key in metrics_avg])} | MSE: {MSE_avg}")
            print(f"Epoch {epoch + 1}/{epochs} | Train metrics: {' | '.join([f'{key}: {metrics_avg[key]}' for key in metrics_avg])} | MSE: {MSE_avg}")

        self.logging.info("Training completed.")
        print("Training completed.")

    def _save_model(self, model):
        """Save the model and its components."""
        if self.save_model:
            model.clap.save_pretrained(self.model_folder)
            model.processor.save_pretrained(self.model_folder)
            torch.save(model.linear.state_dict(), os.path.join(self.model_folder, 'linear.pth'))
            self.logging.info(f"Model saved to {self.model_folder}")
    
    def train_CLAP(self, a, b, batch_size=None, epochs=None, layer_size_1=None, layer_size_2=None, layer_size_3=None, lr=None, save_model=True, stop_shuffle=False):
        # Set default values from self.config if not provided
        if batch_size is None:
            batch_size = self.config.get('batch_size')
        if epochs is None:
            epochs = self.config.get('epochs')
        self.layer_size_1=layer_size_1
        self.layer_size_2=layer_size_2
        self.layer_size_3=layer_size_3
    
        """Main training loop."""
        self.device = self._initialize_device()
    
        self.a=round(a, 1)
        self.b=round(b, 1)
        self.batch_size=batch_size
        self.epochs=epochs
        self._create_folder()
        self._setup_logging()
        self.save_model= save_model
        # Your training logic here
        print(f"Batch size: {batch_size}, Epochs: {epochs}, Stop shuffle: {stop_shuffle}, Save model: {save_model}")
        self.logging.info(f"Batch size: {batch_size}, Epochs: {epochs}, Stop shuffle: {stop_shuffle}, Save model: {save_model}")

        dataloader_train, dataloader_val, dataloader_test = self._prepare_dataloaders(
            batch_size=batch_size,
            stop_shuffle=stop_shuffle
        )

        # print("learning rate: ",float(self.config.get('lr')) )
        # self.logging.info("learning rate: ",float(self.config.get('lr')) )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        self._train_model(
            model=self.model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=self.config.get('epochs'),
            patience=self.config.get('patience'),
            similarity_matrices=self.similarity_matrices
        )

        # Evaluate on test set
        self.eval_pytorch_model(self.best_model, dataloader_test, self.similarity_matrices, "Test")
