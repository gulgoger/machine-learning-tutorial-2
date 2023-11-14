import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('satislar.csv')

# %% veri ön işleme
aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)


# %% data train - test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33,random_state=0)

"""
#%% öznitelik ölçeklendirme -- verileri 0 ila 1 arası veya birbirine yakın değerler olarak yeniden düzenliyor

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

#%% linear regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

# hata alınmasının nedeni eski sürümde çalışırken yeni sürümde indexin yerinin değişmesi


































