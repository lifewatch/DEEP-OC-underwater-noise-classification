import numpy as np

import matplotlib.pyplot as plt

def custom_growth(x):
    if x < 0.625:
        # Linear growth from 0 to 0.625, reaching a value of 0.2 at x = 0.625
        return (0.2 / 0.625) * x
    elif x <= 1:
        # Exponential growth from 0.625 to 1, reaching a value of 1 at x = 1
        a = 5  # Adjust this parameter to control the steepness of the exponential growth
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - 0.625))) / (1 - np.exp(-a * (1 - 0.625)))
    else:
        # Beyond x = 1, keep the function constant at 1
        return 1

list_cat= ['0-1 km', '1-2 km', '2-3 km', '3-4 km', '4-5 km', '5-6 km', '6-7 km', '7-8 km', '8-9 km', '9-10 km', '10+ km']

ids={lbl: i for i, lbl in enumerate(list_cat)}
classes = ids 

# Create a matrix to hold the similarity values
num_classes = len(classes)
similarity_matrix = np.zeros((num_classes, num_classes))
    
for i, class_i in enumerate(classes):
        distance_i  = float(class_i.split('-')[0].replace("-", ".").replace("+ km", ""))

        # (x.split('km')[0].split('_')[-1].replace("-", "."))
        for j, class_j in enumerate(classes):
            distance_j = float(class_j.split('-')[0].replace("-", ".").replace("+ km", ""))
            # print(distance_i, distance_j)
            distance_similarity = 1 - abs(distance_i - distance_j)/10 

            distance_similarity = custom_growth(distance_similarity)

            similarity_matrix[i, j] = distance_similarity
from matplotlib.colors import LinearSegmentedColormap

blues = plt.get_cmap('Blues')
blues_lighter = LinearSegmentedColormap.from_list('Blues_lighter', blues(np.linspace(0, 0.8, 256)))

plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap=blues_lighter, interpolation='nearest')
plt.colorbar(label='Similarity')
plt.xticks(range(num_classes), list_cat, rotation=45)
plt.yticks(range(num_classes), list_cat)
plt.title('Similarity Matrix Visualization for $f(x)$')



# Add similarity values inside the cells
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha='center', va='center', color='black')
save_path = r'C:\Users\wout.decrop\OneDrive - VLIZ\Documents\Papers\Vessel_detection\Figures\model\similarity_matrix.png'
plt.savefig(save_path, dpi=350, bbox_inches="tight")
plt.show()

# plt.show()

# def exponential_growth(x):
#     return x ** 2

# # Generate sample data
# x_values = np.linspace(0, 1, 500)
# y_custom_values = np.array([custom_growth(x) for x in x_values])
# y_exponential_values = exponential_growth(x_values)

# # Specific points to highlight (0.1, 0.2, ..., 0.9)
# highlight_x = np.arange(0.1, 1.0, 0.1)
# highlight_y_custom = np.array([custom_growth(x) for x in highlight_x])
# highlight_y_exponential = exponential_growth(highlight_x)

# # Plot the functions
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, y_custom_values, label='Custom Growth Function', color='blue')
# plt.plot(x_values, y_exponential_values, label='$x^2$ Exponential Function', color='green', linestyle='--')
# plt.scatter(highlight_x, highlight_y_custom, color='blue')
# plt.scatter(highlight_x, highlight_y_exponential, color='green')

# # Adjusted text positions
# offset = 0.05
# for i, (x, y) in enumerate(zip(highlight_x, highlight_y_custom)):
#     if x < 0.6:
#         plt.text(x, y - offset, f'({x:.1f}, {y:.2f})', fontsize=8, color='blue')
#     else:
#         plt.text(x, y + offset, f'({x:.1f}, {y:.2f})', fontsize=8, color='blue')

# for i, (x, y) in enumerate(zip(highlight_x, highlight_y_exponential)):
#     if x < 0.6:
#         plt.text(x, y + offset, f'({x:.1f}, {y:.2f})', fontsize=8, color='green')
#     else:
#         plt.text(x, y - offset, f'({x:.1f}, {y:.2f})', fontsize=8, color='green')

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Custom Growth Function vs. $x^2$ Exponential Function')
# plt.legend()
# plt.grid(True)

# save_path = r'C:\Users\wout.decrop\OneDrive - VLIZ\Documents\Papers\Vessel_detection\Figures\model\costum_function.png'
# plt.savefig(save_path, dpi=350, bbox_inches="tight")
# plt.show()
