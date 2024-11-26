import ROOT
from ROOT import TMath
import math
import sys
from analyze_pulse import pulse_analyzer
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

#--------------------------------------------Settings----------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# Function to plot a triangle given vertices
def plot_triangle(vertices, ax):
    vertices = np.array(vertices)
    triangle = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
    ax.add_patch(triangle)
    ax.plot(vertices[:, 0], vertices[:, 1], 'bo')  # Plot vertices as blue dots

# Create a figure and axis
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Set limits and aspect ratio
for ax in axs.flatten():
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

# Define angles for the triangles (0 to pi/2)
angles = np.linspace(0, np.pi/2, 9)

# Plot triangles in each subplot
for i, angle in enumerate(angles):
    vertices = [[0, 0],
                [np.cos(angle), np.sin(angle)],
                [np.cos(angle), 0]]
    plot_triangle(vertices, axs.flatten()[i])

# Remove extra subplots if necessary
if len(angles) < 9:
    for ax in axs.flatten()[len(angles):]:
        ax.axis('off')

# Save the plot as an image file
plt.tight_layout()
plt.savefig('unit_circle_triangles.png')
plt.show()




"""

def counter(n):
    if n == 2:
        return 1
    else:
        if n % 2 == 0:
            return counter(n // 2) + 1
        else:
            return counter(n - 1) + 1

# Initialize an empty list to store results
results = []

# Iterate through numbers from 2 to 100 and apply counter function
for n in range(2, 101):
    result = counter(n)
    results.append(result)

# Print the array of minimum cuts
print(results)

"""

"""
# Original positions
P = []
for j in range(1, 11):
    for k in range(1, 11):
        P.append([100*k, 100*j])

# Filtered positions within the specified range
filtered_positions = [pos for pos in P if 200 <= pos[0] < 620 and 270 <= pos[1] < 690]

print(filtered_positions)

# Extract x and y coordinates for plotting
x_coords = [pos[0] for pos in filtered_positions]
y_coords = [pos[1] for pos in filtered_positions]

# Create the plot
plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, c='blue', marker='o')
plt.title('Selected Positions in the Specified Region')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.grid(True)
plt.axvline(x=200, color='red', linestyle='--')
plt.axvline(x=620, color='red', linestyle='--')
plt.axhline(y=270, color='green', linestyle='--')
plt.axhline(y=690, color='green', linestyle='--')
plt.savefig('batman.png')
"""