import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def custom_growth(x, a=5, b=0.625):
    if x < b:
        return (0.2 / b) * x
    elif x <= 1:
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - b))) / (1 - np.exp(-a * (1 - b)))
    else:
        return 1

def exponential_growth(x):
    return x ** 2

x_values = np.linspace(0, 1.0001, 500)
highlight_x = np.arange(0.1, 1.0001, 0.1)

plt.figure(figsize=(20, 10))


# Updated b_values
b_values = np.array([ 0.4, 0.5, 0.6, 0.7])
# b_values = np.array([0.5])

# Updated ranges for a based on selected values
a_ranges = {
    0.4: np.array([-3.0, 1.0]),
    0.5: np.array([-2,-1,0.1,1, 2.0]),
    0.6: np.array([-2.0, 2.0]),
    0.7: np.array([-3.0, 1.0]),
}


# b_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
# a_ranges = {
#     0.3: np.array([-7.0, -3.0, -1.0]),
#     0.4: np.array([-5.0, 1.0]),
#     0.5: np.array([2.0]),
#     0.6: np.array([-2.0, 2.0]),
#     0.7: np.array([-3.0, 1.0, 3.0]),
# }

# Fixed alpha value and line thickness
alpha_value = 0.8
line_thickness = 3.0

# Create a list to store legend handles
legend_handles = []

# Iterate over b and corresponding a values
for b in b_values:
    a_range = a_ranges.get(np.round(b, 1), np.array([5]))  # Default to a=5 if b not in a_ranges

    for a in a_range:
        y_custom_values = np.array([custom_growth(x, a, b) for x in x_values])
        highlight_y_custom = np.array([custom_growth(x, a, b) for x in highlight_x])
        
        # Determine color based on the magnitude of 'a'
        if a > 0:
            normalized_a = (a - 0) / (3 - 0)  # Normalize positive 'a' values (3 is the maximum positive)
            color = cm.Blues(normalized_a)    # Scale from light to dark blue
        else:
            # Adjust normalization for negative 'a' values
            normalized_a = (a + 7) / (0 + 7)  # Normalize negative 'a' values (7 is the maximum negative)
            color = cm.Reds(1 - normalized_a)  # Inverse to get darker shades for more negative values
        
        # Plot using a consistent thick line and color based on 'a'
        line, = plt.plot(x_values, y_custom_values, label=rf'$f(x)$ with $a$={"+ " + str(a) if a > 0 else a}, $b$={round(b, 2)}', 
                         linestyle='-', linewidth=line_thickness, color=color, alpha=alpha_value)  # Set alpha for transparency
        
        # Use scatter for highlighting points
        plt.scatter(highlight_x, highlight_y_custom, color=color, s=20, alpha=alpha_value)

        # Create a legend entry with a fixed size
        legend_handles.append(line)

# Plotting exponential growth for reference
y_exponential_values = exponential_growth(x_values)
highlight_y_exponential = exponential_growth(highlight_x)

# Create a handle for the exponential growth line in the legend
exp_growth_line, = plt.plot(x_values, y_exponential_values, label=f'$g(x)$ (MSE)', color='black', linestyle='-', linewidth=3)
plt.scatter(highlight_x, highlight_y_exponential, color='black', s=80, zorder=5)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

# Increase x and y axis font sizes
plt.xlabel('x', fontsize=20)  # Increased font size for x-axis
plt.ylabel('y', fontsize=20)  # Increased font size for y-axis

# Adjusted title position
plt.title('Custom Growth Function for Different a and b Values vs. MSE', fontsize=22, loc='center', pad=20, x=0.65)  # Adjusted title size and position

# Adjust title position more to the right
# plt.text(1.18, 1.05, 'Custom Growth Function for Different a and b Values vs. Exponential Growth', fontsize=28, ha='center')

# Increase the tick label sizes
plt.tick_params(axis='both', which='major', labelsize=20)  # Increased tick label sizes

# Create a custom legend with consistent size
legend_labels = [rf'$f(x)$ with $a$={" " + str(a) if a > 0 else a}, $b$={round(b, 2)}' for b in b_values for a in a_ranges.get(np.round(b, 1), np.array([5]))]
plt.legend(legend_handles + [exp_growth_line], legend_labels + [f'$g(x)$ MSE'], fontsize=20, loc='upper left', bbox_to_anchor=(1, 1), frameon=True, handlelength=2, borderaxespad=1)

# Ensure layout fits the legend inside the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the plot
save_path = fr'C:\Users\wout.decrop\OneDrive - VLIZ\Documents\Papers\Vessel_detection\Figures\loss_function\loss_functions.png'
plt.savefig(save_path, dpi=350, bbox_inches="tight")

# plt.show()
