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

b_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
a_ranges = {
    0.3: np.array([-7.0, -3.0, -1.0]),
    0.4: np.array([-5.0, 1.0]),
    0.5: np.array([2.0]),
    0.6: np.array([-2.0, 2.0]),
    0.7: np.array([-3.0, 1.0, 3.0]),
}

# Define line styles and thickness based on 'b', reversed order
line_styles_thickness = {
    0.3: {'linestyle': '-', 'linewidth': 1.0},    # Thin solid line
    0.4: {'linestyle': '-.', 'linewidth': 1.5},   # Dash-dot normal thickness
    0.5: {'linestyle': '-', 'linewidth': 1.5},    # Normal solid line
    0.6: {'linestyle': '--', 'linewidth': 2.0},   # Dashed normal thickness
    0.7: {'linestyle': '-', 'linewidth': 3.0}     # Thick solid line
}

# Iterate over b and corresponding a values
for b in b_values:
    a_range = a_ranges.get(np.round(b, 1), np.array([5]))  # Default to a=5 if b not in a_ranges
    style_thickness = line_styles_thickness.get(np.round(b, 1), {'linestyle': '-', 'linewidth': 1.5})  # Get line style and thickness based on 'b'
    
    # Calculate alpha value inversely related to b with a minimum value
    alpha_value = max(0.5, 1 - (b - 0.3) / (0.7 - 0.3))  # Ensure alpha is at least 0.5

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
        
        # Plot using line style, thickness, and color based on 'a' and 'b'
        plt.plot(x_values, y_custom_values, label=rf'$f(x)$ with $a$={a}, $b$={round(b, 2)}', 
                 linestyle=style_thickness['linestyle'], linewidth=style_thickness['linewidth'], 
                 color=color, alpha=alpha_value)  # Set alpha for transparency
        
        # Use scatter for highlighting points
        plt.scatter(highlight_x, highlight_y_custom, color=color, s=20, alpha=alpha_value)

# Plotting exponential growth for reference
y_exponential_values = exponential_growth(x_values)
highlight_y_exponential = exponential_growth(highlight_x)

plt.plot(x_values, y_exponential_values, label='$g(x)$', color='black', linestyle='-', linewidth=3)
plt.scatter(highlight_x, highlight_y_exponential, color='black', s=80, zorder=5)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Custom Growth Function for Different a and b Values vs. Exponential Growth', fontsize=16)

# Ensure layout fits the legend inside the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
