# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:28:33 2023
@author: bilgi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load GPS data from a file
def load_gps_data(filepath):
    return pd.read_excel(filepath)

# Function to calculate the distance between consecutive GPS points
def calculate_distance(df):
    df['Distance'] = np.sqrt((df['Latitude'].diff() ** 2) + (df['Longitude'].diff() ** 2))
    df['Distance'] = df['Distance'].fillna(0).cumsum()
    return df

# Function to visualize the GPS track speed profile
def visualize_data(df):
    plt.subplot(1, 2, 1)
    plt.plot(df['Distance'], df['Speed'], color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (m/s)')
    plt.title('GPS Track Speed Profile')

# Function to get and check speed values
def get_and_check_speed(df):
    speed = np.array(df['Speed']) * 3.6  # convert to km/h 
    if len(speed) > 0:
        return speed
    else:
        print('No valid speed data found.')
        return None

# Function to visualize the speed profile
def visualize_speed(speed):
    plt.subplot(1, 2, 2)
    plt.plot(speed, color='blue')
    plt.xlabel('Distance (km)')
    plt.ylabel('Speed (km/h)')
    plt.title('GPS Track Speed Profile')

# Main function to run the program
def main():
    df = load_gps_data('GpsDataSet.xlsx')
    df = calculate_distance(df)
    visualize_data(df)
    speed = get_and_check_speed(df)
    if speed is not None:
        visualize_speed(speed)
    plt.tight_layout()
    plt.show()

# Execute main function
if __name__ == "__main__":
    main()
    

