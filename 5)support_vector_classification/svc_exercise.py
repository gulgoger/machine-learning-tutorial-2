import pandas as pd

data = pd.read_csv("voice.csv")

label = data.iloc[:,-1:].values

from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(data.iloc[:,:-1], label,test_size=0.33,random_state=0)

from sklearn.svm import SVC

svc = SVC(kernel="poly")

svc.fit(x_train,y_train)

result = svc.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, result)
print(cm)

#başarı oranı
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print(accuracy)





































