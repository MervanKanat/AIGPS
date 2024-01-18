# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:37:22 2023
@author: bilgi
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Read the file
df = pd.read_excel('GpsDataSet.xlsx')


# RDP Simplification function
def rdp_simplify(coords, epsilon):
    if len(coords) < 3:
        return coords

    simplified = [coords[0]]  # Keep the first point
    
    max_distance = 0
    max_index = 0
    
    # Check points in the range
    for i in range(1, len(coords) - 1):
        distance = perpendicular_distance(coords[i], coords[0], coords[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # If the farthest point is greater than epsilon, split into two segments and repeat the algorithm for each segment
    if max_distance > epsilon:
        simplified += rdp_simplify(coords[:max_index+1], epsilon)
        simplified += rdp_simplify(coords[max_index:], epsilon)
    else:
        simplified.append(coords[-1])  # If the farthest point is less than epsilon, add the last point and finish
    
    return simplified

def perpendicular_distance(point, line_start, line_end):
    # Calculate the vector representing the line segment
    line_vector = line_end - line_start
    
    # Calculate the vector from the line start to the given point
    point_vector = point - line_start
    
    # Calculate the length of the line segment
    line_length = np.linalg.norm(line_vector)
    
    # Calculate the unit vector in the direction of the line segment
    line_unit_vector = line_vector / line_length
    
    # Calculate the projection of the point vector onto the line unit vector
    projection = np.dot(point_vector, line_unit_vector)
    
    # Check if the projection is outside the line segment (before the start point)
    if projection <= 0:
        return np.linalg.norm(point_vector)
    
    # Check if the projection is outside the line segment (after the end point)
    if projection >= line_length:
        return np.linalg.norm(point - line_end)
    
    # Calculate the perpendicular vector from the point vector to the line segment
    perpendicular = point_vector - projection * line_unit_vector
    
    # Calculate the perpendicular distance as the norm of the perpendicular vector
    return np.linalg.norm(perpendicular)
# Get Latitude and Longitude data
coords = df[['Longitude', 'Latitude']].values

# Epsilon value (simplification level) try-->1
epsilon = 0.005

# Apply the Ramer-Douglas-Peucker algorithm
simplified_coords = rdp_simplify(coords, epsilon)

# Plot the original coordinates
plt.plot(coords[:, 0], coords[:, 1], 'ro-', label='Original')

# Plot the simplified coordinates
simplified_coords = np.array(simplified_coords)  # Convert to a Numpy array
plt.plot(simplified_coords[:, 0], simplified_coords[:, 1], 'bh-', label='Simplified')

# Print the reduction in points
print("{} points reduced to {}!".format(coords.shape[0], simplified_coords.shape[0]))

# Scatter plot of the data points
plt.scatter(df['Longitude'].values, df['Latitude'].values, color='red')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Data')

# Display the legend
plt.legend()

# Show the plot
plt.show()

