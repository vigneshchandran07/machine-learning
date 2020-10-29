# Import required libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

input_x = 'Aerosols'

# Read the CSV file :
data = pd.read_csv("climate_change.csv")
data.head()
# Let's select some features to explore more :
# data = data[[input_x, "CH4", "N2O", "CFC-11", "CFC-12", "TSI", "Aerosols", "Temp"]]
data = data[[input_x, "Temp"]]
# Temp vs CO2:
plt.scatter(data["Temp"], data[input_x], color="blue")
plt.xlabel(input_x)
plt.ylabel("TEMP")
plt.show()
# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data) * 0.8)))]
test = data[(int((len(data) * 0.8))):]
# Modeling:
# Using sklearn package to model data :
regression = linear_model.LinearRegression()
train_x = np.array(train[[input_x]])
train_y = np.array(train[["Temp"]])
regression.fit(train_x, train_y)

# The coefficients:
print("coefficients: ", regression.coef_)  # Slope
print("Intercept: ", regression.intercept_)  # Intercept

# Plotting the regression line:

plt.scatter(train[input_x], train["Temp"], color='blue')
plt.plot(train_x, regression.coef_ * train_x + regression.intercept_, '-r')
plt.xlabel(input_x)
plt.ylabel("TEMP")
plt.show()


# Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features, intercept, slope):
    predicted_values = input_features * slope + intercept
    return predicted_values


# Predicting emission for future car:
future_co2_emission = 400
esitmated_temp = get_regression_predictions(future_co2_emission, regression.intercept_[0], regression.coef_[0][0])
print("Estimated Temperature:", esitmated_temp)
# Checking various accuracy:
from sklearn.metrics import r2_score

test_x = np.array(test[[input_x]])
test_y = np.array(test[['Temp']])
test_y_ = regression.predict(test_x)
print("Mean absolute error: % .2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares(MSE): % .2f" % np.mean((test_y_ - test_y) ** 2))
print("R2 - score: % .2f" % r2_score(test_y, test_y_))
