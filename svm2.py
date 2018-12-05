from sklearn import datasets			# To Get iris dataset
import time
from sklearn.svm import SVC   			# To fit the svm classifier
import numpy as np
#import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#dataset=datasets.load_wine()
#dataset=datasets.load_iris()
dataset=datasets.load_breast_cancer()
#dataset =datasets.load_diabetes()
C=1

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


print" gaussiaan::"
t0=time.time()
classifier = SVC(kernel = 'rbf', random_state = 0,gamma = 0.9)
classifier.fit(X_train, y_train)
print "training time:", round(time.time()-t0,5), "s"


#Predicting the Test Set
t0=time.time()
y_pred = classifier.predict(X_test)
print "Testing time:", round(time.time()-t0,5), "s"
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3)
accuracies.mean()
print"Train Accuracy::",accuracies.mean()*100
accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test,cv = 3)
accuracies.mean()
print"Test Accuracy::",accuracies.mean()*100
#print"--------------------------"
#print y_pred

print" Polynomial::"
t0=time.time()
classifier = SVC(kernel = 'poly', random_state = 0,gamma = 0.9,C=C)
classifier.fit(X_train, y_train)
print "training time:", round(time.time()-t0,5), "s"


#Predicting the Test Set
t0=time.time()
y_pred = classifier.predict(X_test)
print "Testing time:", round(time.time()-t0,5), "s"
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3)
accuracies.mean()
print"Train Accuracy::",accuracies.mean()*100
accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test,cv = 3)
accuracies.mean()
print"Test Accuracy::",accuracies.mean()*100
