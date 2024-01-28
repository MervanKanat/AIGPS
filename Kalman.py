# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:46:05 2023
"""

import pandas as pd
import numpy as np
import openpyxl
import warnings
warnings.filterwarnings("ignore")

# Read the file
df = pd.read_excel('GpsDataSet.xlsx')


df = np.array(df[1:])  # Convert the data to a NumPy array and skip the header

measurements = df[:, :3].astype(float)  # Create the measurements matrix


# State transition matrix (F)
F = np.array([[1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

# Observation matrix (H)
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

# Observation noise covariance matrix (R)
R = np.diag([1e-4, 1e-4, 100])**2

# Initial state mean vector
initial_state_mean = np.hstack([measurements[0, :], 3*[0.]])

# Initial state covariance matrix
initial_state_covariance = np.diag([1e-4, 1e-4, 50, 1e-6, 1e-6, 1e-6])**2

class KalmanFilter:
    def __init__(self, initial_state_mean, initial_state_covariance, F, H, R):
        self.state_mean = initial_state_mean
        self.state_covariance = initial_state_covariance
        self.F = F
        self.H = H
        self.R = R

    def filter(self, measurements):
        filtered_states = []
        for measurement in measurements:
            predicted_state_mean = self.F @ self.state_mean
            predicted_state_covariance = self.F @ self.state_covariance @ self.F.T
            innovation = measurement - self.H @ predicted_state_mean
            innovation_covariance = self.H @ predicted_state_covariance @ self.H.T + self.R
            kalman_gain = predicted_state_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)

            self.state_mean = predicted_state_mean + kalman_gain @ innovation
            self.state_covariance = predicted_state_covariance - kalman_gain @ self.H @ predicted_state_covariance

            filtered_states.append(self.state_mean)

        return np.array(filtered_states)

# Kalman Filter application
kf = KalmanFilter(initial_state_mean, initial_state_covariance, F, H, R)
filtered_states = kf.filter(measurements)

workbook = openpyxl.Workbook()
worksheet = workbook.active
# Printing the filtered states
for i, state in enumerate(filtered_states):
    print(f"Time step {i+1}: {state[:2]}")
    worksheet.append([i+1] + list(state[:2]))

workbook.save("Kalman.xlsx")
