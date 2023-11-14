
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

#standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Svm
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# k fold cross validation
from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator = classifier, X=X_train,y=y_train, cv=4)

print(basari.mean())
print(basari.std())

# grid search
from sklearn.model_selection import GridSearchCV
p = [{"C":[1,2,3,4,5],"kernel":["linear"]},
     {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[1,0.5,0.1,0.01,0.001]}]

gs = GridSearchCV(estimator=classifier,param_grid=p, scoring="accuracy",cv=10, n_jobs=-1)

grid_search = gs.fit(X_train,y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)






















