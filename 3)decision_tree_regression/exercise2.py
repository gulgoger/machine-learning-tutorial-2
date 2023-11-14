# %% Decision tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("maaslar.csv")

egitim = data.iloc[:,1:2].values
maas = data.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(egitim, maas)

predict = dtr.predict(egitim)

plt.scatter(egitim, maas, color="red")
plt.plot(egitim, predict, color="blue")
plt.show()


#%% random forest

data = pd.read_csv("maaslar.csv")

egitim = data.iloc[:,1:2].values
maas = data.iloc[:,2:3].values

from sklearn.ensemble import RandomForestRegressor
# random_state modelin çıkışını çoğaltılamaz hale getirir yani random_state değeri belli olduğunda aynı parametreler ve aynı eğitim verisi verilmişse, aynı sonuçlar üretecektir.
# n_estimators = kaç tane decision tree oluşturulacağı
rfg = RandomForestRegressor(n_estimators=10,random_state=0)

rfg.fit(egitim,maas)

plt.scatter(egitim,maas,color="red")
plt.plot(egitim,predict,color="blue")
plt.show()

















