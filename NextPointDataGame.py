# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:53:27 2023

@author: bilgi
"""
import folium
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Read the file
df = pd.read_excel('GpsDataSet.xlsx')


# Features and targets
features = ['Latitude', 'Longitude', 'Altitude', 'Accuracy', 'Speed', 'Distance']
targets = ['Latitude', 'Longitude']

# Split the data into training and test sets
train_df = df.iloc[:-1]
test_df = df.iloc[-1:]

# Create a linear regression model
model = LinearRegression()

# Create a multi-output regressor
wrapper = MultiOutputRegressor(model)

# Train the model
wrapper.fit(train_df[features], train_df[targets])

# Make predictions
next_lat_lon = wrapper.predict(test_df[features].values.reshape(1, -1))
print("Estimated latitude and longitude for the next point: ", next_lat_lon[0])

# Create a map
map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)

# Add markers to the map for each data point
for index, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']]).add_to(map)
    folium.Marker(next_lat_lon[0], icon=folium.Icon(color='red'), popup='Next Point').add_to(map)
# Show the map
map.save('NextPoint.html')

# GPS Track visualization
latitude = df['Latitude']
longitude = df['Longitude']

# Create a figure with the specified size
plt.figure(figsize=(10, 6))

# Plot the GPS track as a line with blue color and linewidth of 1
plt.plot(longitude, latitude, color='blue', linewidth=1)

# Scatter plot the individual GPS points as red dots
plt.scatter(longitude, latitude, color='red', s=10)

# Scatter plot the next predicted point as a green 'd' marker
plt.scatter(next_lat_lon[0][1], next_lat_lon[0][0], color='green', marker='d', s=100, label='Next Point')

# Set the labels for the x and y axes
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Set the title of the plot
plt.title('GPS Track')

# Show the legend which includes the label 'Next Point'
plt.legend()

# Add a grid to the plot
plt.grid(True)

# Display the plot
plt.show()

