import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(‘Student-Pass-Fail-Data.csv’)
df.head()

x = df.drop(‘Pass_Or_Fail’,axis = 1)
y = df.Pass_Or_Fail
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

y_pred = logistic_regression.predict(x_test)

#accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage

# predicting whether a student will fail or pass
First_Friend = logistic_regression.predict((np.array([4, 38]).reshape(1, -1)))
First_Friend # Failed

Second_Friend = logistic_regression.predict((np.array([8, 29]).reshape(1, -1)))
Second_Friend # Passed

