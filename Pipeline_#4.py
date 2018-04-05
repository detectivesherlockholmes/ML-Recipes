'''
Created on 01-Apr-2018

@author: sherlock
'''
from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.neighbors.classification import KNeighborsClassifier
iris = load_iris()

X= iris.data
y= iris.target

X_train ,X_test , y_train, y_test = train_test_split(X,y, test_size= .5)

# my_classifier = tree.DecisionTreeClassifier()
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

print(accuracy_score(y_test,predictions))