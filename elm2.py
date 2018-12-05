import numpy as np
from sklearn import datasets	
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

dataset=datasets.load_breast_cancer()

print " data set Description :: ", dataset['DESCR']
print "feature data :: ", dataset['data']
print "target :: ", dataset['target']


X = dataset.data[:,:]  
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(y)

#feature scaling'
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


nh = 10
t0=time.time()
srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')

classifier=GenELMClassifier(hidden_layer=srhl_tanh)

classifier.fit(X_train, y_train)
print "training time:", round(time.time()-t0,5), "s"

t0=time.time()
y_pred = classifier.predict(X_test)
print "Testing time:", round(time.time()-t0,5), "s"
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
print"Train Accuracy::",accuracies.mean()*100
accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test,cv = 3)
accuracies.mean()
print"Test Accuracy::",accuracies.mean()*100