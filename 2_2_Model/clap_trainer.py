import os
import yaml
import torch
from dataset import CLAP_Vessel_Distance
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Load configuration file
config_path = os.path.join('..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define a training function for Ray Tune
def train_fn(config, checkpoint_dir=None):
    # Extract hyperparameters from the config passed by Ray Tune
    a = config["a"]
    b = config["b"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    layer_size_1 = config["layer_size_1"]
    layer_size_2 = config["layer_size_2"]
    layer_size_3 = config["layer_size_3"]
    # Initialize the model with custom layer sizes
    L = CLAP_Vessel_Distance(config)
    
    # Set weights (optional: you can also tune these)
    weights = {
        'distance_weight': 1,
        'speed_weight': 0,
        'activity_weight': 0,
        'vessel_type_weight': 0
    }
    
    # Example optimizer, adapt as needed
    
    # Update model architecture if necessary with the new layer sizes (if applicable)
    # L.update_layer_sizes(layer_size_1, layer_size_2)  # Assume you have a function to modify the model architecture
    
    # Train the model
    L.train_CLAP(a=a, b=b, batch_size=batch_size, epochs=epochs, layer_size_1=layer_size_1, layer_size_2=layer_size_2, layer_size_3=layer_size_3, lr=learning_rate,save_model=True)
    
    # # Report the validation loss to Ray Tune
    # val_loss = L.get_validation_loss()  # This function should return the validation loss or accuracy.
    
    # # Report the validation loss to Ray Tune
    # tune.report(val_loss=val_loss)

# Define the search space for the hyperparameters
def get_search_space():
    return {
        "a": tune.uniform(-3, 3),           # Hyperparameter `a` with a uniform distribution between -3 and 2
        "b": tune.uniform(0.4, 0.7),        # Hyperparameter `b` with a uniform distribution between 0.4 and 0.7
        "batch_size": tune.choice([ 8, 16, 32]),  # Batch size options
        "learning_rate": tune.choice([1e-7,1e-6, 1e-5]), # Log-uniform distribution for learning rate
        "layer_size_1": tune.choice([128, 256, 512]),  # Hidden layer 1 sizes (starting from larger to smaller)
        "layer_size_2": tune.choice([64, 128]),        # Hidden layer 2 sizes (should be smaller than layer 1)
        "layer_size_3": tune.choice([32]), 
        "epochs": 150  # Total number of epochs for the training
    }

# Define the number of trials and search method
num_samples = 200  # Number of trials to run in parallel

# Run hyperparameter search using Ray Tune
scheduler = ASHAScheduler(
    metric="val_mse",  # Metric to optimize
    mode="max",         # Minimize validation loss
    max_t=150,          # Max number of epochs
    grace_period=1,     # Minimum number of epochs to run
    reduction_factor=2, # Reduce resources by factor of 2
)

# Merge the YAML config with the search space
search_space = get_search_space()
merged_config = {**config, **search_space}  # Merging both config and search space

analysis = tune.run(
    train_fn,
    name="tune_vessel_distance",
    config=merged_config,  # Pass the merged configuration to Ray Tune
    num_samples=num_samples,
    scheduler=scheduler,
    resources_per_trial={"cpu": 4, "gpu": 1},  # Adjust the resources for each trial
)

# # After the tuning, retrieve the best configuration
# print("Best configuration found: ", analysis.best_config)

# Optionally, you can access the best trial's results
best_trial = analysis.get_best_trial("val_mse", "max", "all")
print(f"Best trial validation loss: {best_trial.last_result['val_mse']}")

# Assuming `analysis` is the result of `tune.run()`
best_config = analysis.get_best_config(metric="val_mse", mode="max")
print("Best configuration found: ", best_config)

