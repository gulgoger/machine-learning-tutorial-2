import pandas as pd

data = pd.read_csv('Iris.csv')


species = data.iloc[:,-1:].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier sınıfından bir nesne ürettik
# n_neighbors : K değeridir. Bakılacak eleman sayısıdır. Default değeri 5'tir.
# metric : Değerler arasında uzaklık hesaplama formülüdür.
# p : Alternatif olarak p parametreside verilir. p değerini 2 vererek uzaklık hesaplama formülünü
# minkowski yerine öklid olarak değiştirebilirsiniz.
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(x_train,y_train.ravel())

result = knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
# Sonuç : 0.98
print(accuracy)