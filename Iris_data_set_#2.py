'''
Created on 01-Apr-2018

@author: sherlock
'''
from sklearn.datasets import load_iris
from sklearn.tree import tree
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

print(iris.data[0])
print(iris.target[0])

clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

print(clf.predict([[5.1, 3.5, 1.4, 0.19]]))