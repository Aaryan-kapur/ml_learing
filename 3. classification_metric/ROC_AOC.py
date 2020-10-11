#The ROC curve (or receiver operating characteristics) is a valuable tool to compare different classifiers that can assign a score to their predictions. In general, this score can be interpreted as a probability, so it's bounded between 0 and 1.
#the obliqued line goes from (0,0) to (1,1)
# The x axis represents the increasing false positive rate (also known as specificity), while the y axis represents the true positive rate (also known as sensitivity). The dashed oblique line represents a perfectly random classifier, so all the curves below this threshold perform worse than a random choice, while the ones above it show better performances.
# Of course, the best classifier has an ROC curve split into the segments [0, 0] - [0, 1] and [0, 1] - [1, 1], and our goal is to find algorithms whose performances should be as close as possible to this limit. 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
lr = LogisticRegression()
lr.fit(X_train, Y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
Y_scores = lr.decision_function(X_test)

#computing the ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_scores)


#The output is made up of the increasing true and false positive rates and the decreasing thresholds 
#Before proceeding, it's also useful to compute the area under the curve (AUC), whose value is bounded between 0 (worst performances) and 1 (best performances), with a perfectly random value corresponding to 0.5:
from sklearn.metrics import auc
auc(fpr, tpr)


#plotting the ROC curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='red', label='Logistic regression (AUC: %.2f)' % auc(fpr, tp
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()