#the most important metric is the accuracy,

#skicit learn has it inbuilt
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, lr.predict(X_test))

#or

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage


#Another very common approach is based on zero-one loss function
#usually done on normalised scores

from sklearn.metrics import zero_one_loss
#on normalised score
zero_one_loss(Y_test, lr.predict(X_test))
#on non normalised score
zero_one_loss(Y_test, lr.predict(X_test), normalize=False)

