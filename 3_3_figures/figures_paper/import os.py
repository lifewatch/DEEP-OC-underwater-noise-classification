import os
import glob

# Define the base directory
base_dir = r"\\fs\shared\onderzoek\6. Marine Observation Center\Projects\IMAGINE\UC6\plots\plots_per_station_6_paper-smooth_ALL"

image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp", "*.gif")

# Find all image files in subdirectories
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))

# Loop through each image file and print the corresponding LaTeX figure code
for image_path in image_files:
    # break
    # print(image_path)
    # Extract the file name (last part of the path, excluding directories)
    image_name = os.path.basename(image_path)
    if "Grafton" in image_name:
        # Use only the image file name (without the directory part)
    # Use the image file name in the LaTeX \includegraphics statement
        print(f"\\begin{{figure*}}[!t]")
        print("    \\centering")
        print(f"    \\includegraphics[width=\\textwidth]{{figures/supplementary/time_series_distances/grafton/{image_name}}}")  # Correct LaTeX syntax with file name
        print("    \\caption{The top section displays the power spectrum of an entire day's recording, while the bottom section shows the distance between nearby vessels and the hydrophone. Vertical lines in the power spectrum correspond to peaks in vessel proximity. Vessels are colored based on their vessel type.}")
        print("    \\label{fig-AIS_lacking}")
        print("\\end{figure*}")
        print("\n")  # Add a newline between figures for better readability
    else:
                # Use the image file name in the LaTeX \includegraphics statement
        print(f"\\begin{{figure*}}[!t]")
        print("    \\centering")
        print(f"    \\includegraphics[width=\\textwidth]{{figures/supplementary/time_series_distances/gardencity/{image_name}}}")  # Correct LaTeX syntax with file name
        print("    \\caption{The top section displays the power spectrum of an entire day's recording, while the bottom section shows the distance between nearby vessels and the hydrophone. Vertical lines in the power spectrum correspond to peaks in vessel proximity. Vessels are colored based on their vessel type.}")
        print("    \\label{fig-AIS_lacking}")
        print("\\end{figure*}")
        print("\n")  # Add a newline between figures for better readability
