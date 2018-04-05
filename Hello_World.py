'''
Created on 01-Apr-2018

@author: Mayank
'''
# import sklearn
# features = [[140,"smooth"],[130,"smooth"],[150,"bumpy"],[170,"bumpy"]]
# labels = ["apple","apple","orange","orange"]
from sklearn.tree import tree

features = [[140,1],[130,1],[150,0],[170,0]]
labels = [1,1,0,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[160,0]]))