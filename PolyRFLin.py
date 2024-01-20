import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# Data Loading
data = pd.read_excel('GpsDataSet.xlsx')
print(data)
x = data.iloc[:, :1].values
y = data.iloc[:, 1].values

# Define a function to plot regression
def plot_regression(x, y, degree, ax):
     # Create polynomial features
    poly_reg = PolynomialFeatures(degree=degree)
    
    # Transform input features into polynomial features
    x_poly = poly_reg.fit_transform(x)
    
    # Create linear regression model
    lin_reg = LinearRegression()
    
    # Fit the model to polynomial features
    lin_reg.fit(x_poly, y)
    
    # Make predictions
    predictions = lin_reg.predict(x_poly)
    
    # Calculate mean squared error
    error = mean_squared_error(y, predictions)
    
    # Plot scatter plot
    ax.scatter(x, y, color='red')
    
    # Plot regression line
    ax.plot(x, predictions, color='blue')
    
    # Set axis labels and title
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_title(f"Polynomial Regression (degree={degree})\nError: {error}")
    
    # Print predicted value for x=41
    print(f"Polynomial Regression (degree={degree}) 41: {lin_reg.predict(poly_reg.fit_transform([[41]]))}")

# Create a figure with four subplots
fig, axs = plt.subplots(3, 2, figsize=(11, 7))

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
predictions = lin_reg.predict(x)
error = mean_squared_error(y, predictions)
axs[0, 0].scatter(x, y, color='red')
axs[0, 0].plot(x, predictions, color='blue')
axs[0, 0].set_xlabel("Latitude")
axs[0, 0].set_ylabel("Longitude")
axs[0, 0].set_title(f"Linear Regression\nError: {error}")

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x, y.ravel())
predictions = rf_reg.predict(x)  
error_rf = mean_squared_error(y, predictions)
axs[2, 1].scatter(x, y, color='red')
axs[2, 1].plot(x, predictions, color='blue')
axs[2, 1].set_xlabel("Latitude")
axs[2, 1].set_ylabel("Longitude")
axs[2, 1].set_title(f"Random Forest Regression\nError: {error_rf}")


# Polynomial Regressions
degrees = [2, 3, 4, 5]
axs = axs.flatten()[1:]
for degree, ax in zip(degrees, axs):
    plot_regression(x, y, degree, ax)


# Add legends
axs[0].legend(['Data Points','Predictions'], loc='upper right', fontsize='large')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


print(f"               Linear Regression 41: {lin_reg.predict([[41]])}")
print(f"        Random Forest Regression 41: {rf_reg.predict([[41]])}")