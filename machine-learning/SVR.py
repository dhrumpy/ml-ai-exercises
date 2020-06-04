#!/usr/bin/env python
# coding: utf-8


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = np.expand_dims(dataset.iloc[:, 2].values,axis=1)
dataset.head()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

from sklearn.svm import SVR
# kerner = rbf: radial basis function kernel
regressor = SVR(kernel='rbf')
regressor.fit(X=X_scaled,y=y_scaled)
regressor.predict(sc_X.transform(np.array([[6.5]])))

# Visualising the SVR Regression results
plt.scatter(X_scaled, y_scaled, color = 'red', label = 'Data points')
plt.plot(X_scaled, regressor.predict(X_scaled), color = 'blue')
y_pred_scaled = regressor.predict(sc_X.transform(np.array([[6.5]])))
plt.plot(sc_X.transform(np.array([[6.5]])), y_pred_scaled, 's', label='Prediction: SVR')
# plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level (scaled)')
plt.ylabel('Salary (scaled)')
plt.legend()
plt.grid()
plt.show()

# Visualising the SVR Regression results
plt.scatter(X, y, color = 'red', label = 'Data points')
plt.plot(sc_X.inverse_transform(X_scaled), sc_y.inverse_transform(regressor.predict(X_scaled)), color = 'blue')
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
plt.plot([[6.5]], y_pred, 's', label='Prediction: SVR')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()

