import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import random


import matplotlib.cm as cm
# Path to the CSV file
file_path = r'\\fs\SHARED\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\figures\samples_time.csv'
random.seed(42)  # Set a specific seed for reproducibility
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Extract date
df['date'] = df['event_time'].dt.date

df['DayOfYear'] = df['event_time'].dt.dayofyear

# Group data by station, day of year, and deployment
grouped = df.groupby(['station', 'DayOfYear', 'deployment']).size().unstack(fill_value=0)

# Additional dates
grafton_dates = [
    "2022-01-22",
    "2022-02-07",
    "2022-02-03",
    '2022-02-11',
    "2022-02-23",
    "2022-03-15",
    "2022-04-18",
    "2022-08-28",
    "2022-09-15",
    "2022-10-30"
]

gardencity_dates = [
    "2022-01-22",
    "2022-01-26",
    "2022-02-11",
    "2022-02-25",
    "2022-03-13",
    "2022-03-25",
    "2022-04-24",
    "2022-05-04",
    "2022-05-10",
    "2022-05-20"
]

random.shuffle(grafton_dates)
random.shuffle(gardencity_dates)

half_grafton = len(grafton_dates) // 2
half_gardencity = len(gardencity_dates) // 2

grafton_val = grafton_dates[:half_grafton]
grafton_test = grafton_dates[half_grafton:]

gardencity_val = gardencity_dates[:half_gardencity]
gardencity_test = gardencity_dates[half_gardencity:]

desired_pairs_val = [('Grafton', date) for date in grafton_val] + [('GardenCity', date) for date in gardencity_val]
desired_pairs_test = [('Grafton', date) for date in grafton_test] + [('GardenCity', date) for date in gardencity_test]

# Function to create circular plot
def create_year_clock(ax, data, station, desired_pairs_val, desired_pairs_test):
    num_days = 365
    angles = np.linspace(np.pi/2, -3*np.pi/2, num_days, endpoint=False)
    
    # Get unique deployments
    deployments = sorted(df['deployment'].unique())

    
    # Create a categorical colormap with unique colors for each deployment
    cmap = plt.cm.get_cmap('tab20', len(deployments))
    deployment_text_added = set()  # Set to track if text is already added for deployment
    colormap = cm.get_cmap('tab10')
    train_color = colormap(0)  # Blue
    val_color = colormap(2)    # Green
    test_color = colormap(1)   # Orange
    for index, count in data.items():
        deployment, day = index
        if count > 5:
            angle = angles[day - 1]
            
            # Determine color based on training, validation, and test pairs
            date_str = pd.Timestamp('2022') + pd.to_timedelta(day - 1, unit='D')
            date_str = date_str.strftime('%Y-%m-%d')
            pair = (station, date_str)
            if pair in desired_pairs_val:
                color = val_color
            elif pair in desired_pairs_test:
                color = test_color
            else:
                color = train_color
                # color =cmap(deployments.index(15810))
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color=color, linewidth=3.5)

            # Add deployment number as text only once
            # if deployment not in deployment_text_added:
            #     ax.text(1.05 * np.cos(angle), 1.05 * np.sin(angle), str(deployment), fontsize=8, color=color)
            #     deployment_text_added.add(deployment)

    # Add month annotations and grid lines
    # # Add month annotations and grid lines
    for i in range(12):
        month_angle = np.pi/2 - i * (2 * np.pi / 12)
        ax.text(1.15 * np.cos(month_angle), 1.15 * np.sin(month_angle), calendar.month_abbr[i+1], ha='center', va='center')
        ax.plot([0, 1.1 * np.cos(month_angle)], [0, 1.1 * np.sin(month_angle)], color='gray', linestyle='--', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_title(station)
    ax.set_xticks([])
    ax.set_yticks([])

    # Create legend for train, validation, and test
    train_line = plt.Line2D([0], [0], color=train_color, linewidth=3.5, label='Train')
    val_line = plt.Line2D([0], [0], color=val_color, linewidth=3.5, label='Validation')
    test_line = plt.Line2D([0], [0], color=test_color, linewidth=3.5, label='Test')

    ax.legend(handles=[train_line, val_line, test_line], loc='upper left', fontsize='medium')

    return ax

# Create circular plots for each station
stations = grouped.index.get_level_values('station').unique()
fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Larger figure size

for ax, station in zip(axs, stations):
    data = grouped.loc[station].unstack(fill_value=0)
    ax = create_year_clock(ax, data, station, desired_pairs_val, desired_pairs_test)
    
    # Add title to each subplot
    ax.set_title(station)
plt.subplots_adjust(wspace=0.05)  #
save_path = r'C:\Users\wout.decrop\OneDrive - VLIZ\Documents\Papers\Vessel_detection\Figures\data_collection\deployments_over_time.png'

plt.savefig(save_path, dpi=350, bbox_inches="tight")
# plt.show()

