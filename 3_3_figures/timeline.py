import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar

# Path to the CSV file
file_path = r'\\fs\SHARED\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\figures\samples_time.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Extract date
df['date'] = df['event_time'].dt.date

df['DayOfYear'] = df['event_time'].dt.dayofyear

# Group data by station, day of year, and deployment
grouped = df.groupby(['station', 'DayOfYear', 'deployment']).size().unstack(fill_value=0)

# Function to create circular plot
def create_year_clock(ax, data, station):
    num_days = 365
    angles = np.linspace(np.pi/2, -3*np.pi/2, num_days, endpoint=False)
    
    # Get unique deployments
    deployments = sorted(df['deployment'].unique())
    
    # Create a categorical colormap with unique colors for each deployment
    cmap = plt.cm.get_cmap('tab20', len(deployments))
    
    handles = []  # List to store handles for legend
    labels = []   # List to store labels for legend
    added_deployments = set()  # Set to keep track of added deployments

    for index, count in data.items():
        deployment, day = index
        if count > 5:
            angle = angles[day - 1]
            color = cmap(deployments.index(deployment))
            print(deployment, color)
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color=color, linewidth=3.5)

            if deployment not in added_deployments: 
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color))
                labels.append(f'Deployment {deployment}')  # Add "Deployment" prefix to label
                added_deployments.add(deployment)  # Add deployment to the set of added deployments

    # Add month annotations and grid lines
    for i in range(12):
        month_angle = np.pi/2 - i * (2 * np.pi / 12)
        ax.text(1.15 * np.cos(month_angle), 1.15 * np.sin(month_angle), calendar.month_abbr[i+1], ha='center', va='center')
        ax.plot([0, 1.1 * np.cos(month_angle)], [0, 1.1 * np.sin(month_angle)], color='gray', linestyle='--', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_title(station)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, handles, labels

# Create circular plots for each station
stations = grouped.index.get_level_values('station').unique()
fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Larger figure size

for ax, station in zip(axs, stations):
    data = grouped.loc[station].unstack(fill_value=0)
    ax, handles, labels = create_year_clock(ax, data, station)
    ax.legend(handles, labels, loc='upper left', fontsize='large') # Add legend for each station

plt.show()
