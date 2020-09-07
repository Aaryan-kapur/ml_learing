import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#intercept
print(regressor.intercept_)

#slope
print(regressor.coef_)

#to make pred
y_pred = regressor.predict(X_test)

#comparin predicted output with actual

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df