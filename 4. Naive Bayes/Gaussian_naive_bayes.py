#useful when working with continuous values, whose probabilities can be modelled using gaussian distribution.
#necessary to estimate the variance and mean f each of them using the maximum likelihood approach. 

#making a random dataset
from sklearn.datasets import make_classification
nb_samples = 300
X,Y = make_classification(n_samples = nb_samples. n_features = 2, n_informative = 1, n_clusters_per_class=1)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

#training 2 diff models to compare teh classifiers. 
#gaussian Bayes and logistic regressin
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_gnb_score = gnb.predict_proba(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_lr_score = lr.decision_function(X_test)

fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(Y_test, Y_gnb_score[:, 1])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(Y_test, Y_lr_score)

Niave Bayes has a better performance. but both have similiar accuracy under AOC. 