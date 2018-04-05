'''
Created on 01-Apr-2018

@author: sherlock
'''

import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)
class ScrappyKNN():
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self,X_test):
        predictions = []
        for row in X_test:
#             label = random.choice(self.y_train)
            label = self.closest(row)        
            predictions.append(label)
        return predictions
    def closest(self,row):
        best_dis = euc(row, X_train[0])
        best_idx = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dis:
                best_dis = dist
                best_idx  = i
        return self.y_train[best_idx]
        
        

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
# my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

print(accuracy_score(y_test,predictions))

