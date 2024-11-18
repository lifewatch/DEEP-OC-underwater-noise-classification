import numpy as np
import matplotlib.pyplot as plt

def custom_growth(x,a=5,b=0.625):
    if x < b:
        return (0.2 / b) * x
    elif x <= 1:
        # a = 5
        return 0.2 + (1 - 0.2) * (1 - np.exp(-a * (x - b))) / (1 - np.exp(-a * (1 - b)))
    else:
        return 1

def exponential_growth(x):
    return x ** 2

a=5
b=0.5
x_values = np.linspace(0, 1.0001, 500)
y_custom_values = np.array([custom_growth(x,a,b) for x in x_values])
y_exponential_values = exponential_growth(x_values)

highlight_x = np.arange(0.1, 1.0001, 0.1)
highlight_y_custom = np.array([custom_growth(x,a,b) for x in highlight_x])
highlight_y_exponential = exponential_growth(highlight_x)

plt.figure(figsize=(15, 8))
plt.plot(x_values, y_custom_values, label=rf'$f(x)$ with $a$={a}', color='blue', linewidth=2.5)
plt.plot(x_values, y_exponential_values, label='$g(x)$', color='green', linestyle='--', linewidth=2.5)
plt.scatter(highlight_x, highlight_y_custom, color='blue', s=50, zorder=5)
plt.scatter(highlight_x, highlight_y_exponential, color='green', s=50, zorder=5)

# Adding vertical grid lines on x-axis with labels from 0 to 1
for x in np.arange(0, 1.1, 0.1):
    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7)

# Label the x-axis at intervals of 0.1
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)

offset_y = 0.05
offset_x = 0.05
# Adjusted offset for annotations when x >= 0.7
closer_offset_x = 0.015
closer_offset_y = 0.015

# Annotate for custom growth
for x, y in zip(highlight_x, highlight_y_custom):
    if x <= 0.35:
        plt.text(x, y + offset_y, f'({x:.1f}, {y:.2f})', fontsize=10, color='blue', ha='center')
    elif 0.4 <= x <= 0.6:
        plt.text(x, y - offset_y, f'({x:.1f}, {y:.2f})', fontsize=10, color='blue', ha='center')
    elif x >= 0.7 and x<1:
        plt.text(x - closer_offset_x, y + closer_offset_y, f'({x:.1f}, {y:.2f})', fontsize=10, color='blue', ha='right')
# Adjusted offset for annotations when x >= 0.7
closer_offset_x = 0.025
closer_offset_y = 0.025
# Annotate for exponential growth
for x, y in zip(highlight_x, highlight_y_exponential):
    if x <= 0.35:
        plt.text(x, y - offset_y, f'({x:.1f}, {y:.2f})', fontsize=10, color='green', ha='center')
    elif 0.4 <= x <= 0.6:
        plt.text(x, y + offset_y, f'({x:.1f}, {y:.2f})', fontsize=10, color='green', ha='center')
    elif x >= 0.7 and x<1:
        plt.text(x + closer_offset_x, y, f'({x:.1f}, {y:.2f})', fontsize=10, color='green', ha='left')

# # Explicit annotation for the point (1.0, 1.0)
plt.text(1.0, custom_growth(1.0) + offset_y/2, f'(1.00, {custom_growth(1.0):.2f})', fontsize=10, color='blue', ha='center')
plt.text(1.0+ closer_offset_x/5*3, exponential_growth(1.0) - offset_y*3/2, f'(1.00, {exponential_growth(1.0):.2f})', fontsize=10, color='green', ha='center')

plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title(rf'$f(x)$ Function with $a$={a} vs. $g(x)$ Function', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks(fontsize=12)

save_path = fr'C:\Users\wout.decrop\OneDrive - VLIZ\Documents\Papers\Vessel_detection\Figures\model\costum_function_{a}_{b}.png'
# plt.savefig(save_path, dpi=350, bbox_inches="tight")
plt.show()
