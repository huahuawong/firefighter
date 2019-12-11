import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn.metrics as metric
# %matplotlib inline
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC  # "Support Vector Classifier"
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("C:\\Users\\User\\Desktop\\IUPUI Masters\\Fall 2019\\Art Intel\\AI Final Ver\\"
                   "wildfire_data.csv")

X = data.drop('Class', axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(svclassifier, X, y, cv=10, scoring='accuracy')
print(scores)
print("The mean of the r2 score is: ", np.mean(scores))
