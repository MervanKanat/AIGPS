# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:10:31 2023

@author: bilgi
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

def fit_ols_model(df, column_to_drop=None):
    # Drop the specified column if provided
    if column_to_drop:
        df = df.drop(column_to_drop, axis=1)
    
    # Prepare the feature matrix (X) and the target variable (y)
    X = np.append(arr=np.ones((len(df), 1)), values=df.iloc[:, :-1], axis=1)
    y = df.iloc[:, -1]

    # Fit the Ordinary Least Squares (OLS) regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Obtain the summary of the regression results
    summary = results.summary()
    
    # Extract the second table from the summary (which contains the coefficient results)
    summary_without_notes = summary.tables[1]
    
    # Extract the Adjusted R-squared value
    adj_r_squared = results.rsquared_adj
    
    return summary_without_notes, adj_r_squared


# Read the dataset from 'GpsDataSet.xlsx' file
df = pd.read_excel('GpsDataSet.xlsx')

# Fit the model and print the summary
summary1, adj_r_squared1 = fit_ols_model(df)

# Fit the model after dropping a column and print the summary
summary2, adj_r_squared2 = fit_ols_model(df, column_to_drop=df.columns[3])

# Print the regression results and Adjusted R-squared values
print("\nRegression Results (Original Data):\n", summary1, "\nAdjusted R-squared:", adj_r_squared1, "\n")
print("Regression Results (After Column Drop):\n", summary2, "\nAdjusted R-squared:", adj_r_squared2)
