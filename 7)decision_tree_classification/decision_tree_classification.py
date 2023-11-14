import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')

print(veriler)

x = veriler.iloc[:,1:4].values  # bağımsız değişken
y = veriler.iloc[:,4:].values    # bağımlı değişken
print(y)

#%% train- test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# data scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
print(x_train)

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


#%% CONFUSION MATRİX

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% KNN 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


#%% SVC

from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)   
print("SVC")
print(cm)

#%% NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)    


cm = confusion_matrix(y_test, y_pred)   
print("GNB")
print(cm)

#%% DECISION TREE CLASSIFICATION

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(x_test, y_pred)   
print("DTC")
print(cm)
















































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')

print(veriler)

x = veriler.iloc[:,1:4].values  # bağımsız değişken
y = veriler.iloc[:,4:].values    # bağımlı değişken
print(y)

#%% train- test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# data scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)




#%% CONFUSION MATRİX

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% KNN 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)





























































































































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')

print(veriler)

x = veriler.iloc[:,1:4].values  # bağımsız değişken
y = veriler.iloc[:,4:].values    # bağımlı değişken
print(y)

#%% train- test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# data scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)




#%% CONFUSION MATRİX

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% KNN 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)






































































