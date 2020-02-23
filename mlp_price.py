from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score

scaler = StandardScaler() 
a = np.loadtxt('Book2.csv',delimiter=',')
X = a[0:15,:-1]
scaler.fit(X)
X = scaler.transform(X)
y = a[0:15,-1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(150, ), random_state=1)
clf.fit(X, y)
#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
#              solver='lbfgs')
test=a[16:17,:-1]
test = scaler.transform(test)
ans=clf.predict(test)


