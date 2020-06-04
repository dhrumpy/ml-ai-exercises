#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('/data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
dataset.head()

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicting a new result with Linear Regression
y_1 = lin_reg.predict([[6.5]])
# Predicting a new result with Polynomial Regression
y_2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(y_1, y_2)


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(6.5, y_1, 's', label='Prediction: Linear regression')
plt.plot(6.5, y_2, 's', label='Prediction: Polynomial regression')
plt.scatter(X, y, color = 'red', label='Data points')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.grid()
plt.legend()
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
