#method 1 using seaborn
#input the model in format()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format()
plt.title(all_sample_title, size = 15);


#method 2 using skicit
#cm = confusion_matrix(y_true=Y_test, y_pred=lr.predict(X_test)) from X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=Y_test, y_pred=lr.predict(X_test))
cm[::1, ::-1]