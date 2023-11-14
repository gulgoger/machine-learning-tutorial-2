
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
veriler = pd.read_csv('Wine.csv')
print(veriler)

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

#train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)


#standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#principal component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)


X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# pca dönüşümünden önce gelen logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# pca dönüşümünden sonra gelen logistic regression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)


y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

# actual / PCA olmadan çıkan sonuç
print("gerçek / PCAsiz")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# actual / PCA sonrası çıkan sonuç
print("gerçek / PCA ile")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

# PCA sonrası / PCA öncesi
print("PCAsiz ve PCAli")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)


# linear discriminant analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# lda dönüşümünden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)


# lda verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

# LDA sonrası / orijinal
print("LDA ve orijinal")
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)




















































