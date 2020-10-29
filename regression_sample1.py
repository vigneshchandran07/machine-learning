# Import required libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Read the CSV file :
data = pd.read_csv("FuelConsumptionCo2.csv")
data.head()
# Let's select some features to explore more :
data = data[["ENGINESIZE", "CO2EMISSIONS"]]
# ENGINESIZE vs CO2EMISSIONS:
plt.scatter(data["ENGINESIZE"], data["CO2EMISSIONS"], color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data) * 0.8)))]
test = data[(int((len(data) * 0.8))):]
# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[["ENGINESIZE"]])
train_y = np.array(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)
# The coefficients:
print("coefficients: ", regr.coef_)  # Slope
print("Intercept: ", regr.intercept_)  # Intercept

# Plotting the regression line:

plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color='blue')
plt.plot(train_x, regr.coef_ * train_x + regr.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features, intercept, slope):
    predicted_values = input_features * slope + intercept
    return predicted_values


# Predicting emission for future car:
my_engine_size = 6
estimatd_emission = get_regression_predictions(my_engine_size, regr.intercept_[0], regr.coef_[0][0])
print("Estimated Emission:", estimatd_emission)
# Checking various accuracy:
from sklearn.metrics import r2_score

test_x = np.array(test[['ENGINESIZE']])
test_y = np.array(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: % .2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares(MSE): % .2f" % np.mean((test_y_ - test_y) ** 2))
print("R2 - score: % .2f" % r2_score(test_y_, test_y))
