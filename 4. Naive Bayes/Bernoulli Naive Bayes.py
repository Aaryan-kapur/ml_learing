#If X is random variable and is Bernoulli-distributed, it can assume only two values (for simplicity, let's call them 0 and 1) and their probability is:

# P(X) = {p if x = 1
#         q if x = 0
# where q = 1-p md 0 < p < 1

#To try this algorithm with scikit-learn, we're going to generate a dummy dataset. Bernoulli naive Bayes expects binary feature vectors; however, the class "BernoulliNB" has a binarize parameter, which allows us to specify a threshold that will be used internally to transform the features:

#making a random dataset
from sklearn.datasets import make_classification
nb_samples = 300
X,Y = make_classification(n_samples = nb_samples. n_features = 2, n_informative = 1, n_clusters_per_class=1)

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, Y_train)
bnb.score(X_test, Y_test)

#predictions
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
bnb.predict(data)

