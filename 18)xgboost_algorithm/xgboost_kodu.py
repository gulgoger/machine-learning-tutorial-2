import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)











