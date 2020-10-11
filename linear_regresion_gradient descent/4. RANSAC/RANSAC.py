#some times a linear regression model has outliers which cz the line to move a lot. to help stop that we use RANSAC

from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(X.reshape((-1, 1)), Y.reshape((-1, 1)))
lr.intercept_

lr.coef_

from sklearn.linear_model import RANSACRegressor
rs = RANSACRegressor(lr)
rs.fit(X.reshape((-1, 1)), Y.reshape((-1, 1)))
rs.estimator_.intercept_

rs.estimator_.coef_
