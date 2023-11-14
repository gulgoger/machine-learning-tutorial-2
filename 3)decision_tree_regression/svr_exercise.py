
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("maaslar.csv")

egitim = data.iloc[:,1:2].values

maas = data.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler

sc_egitim = StandardScaler()

egitim_scale = sc_egitim.fit_transform(egitim)

sc_maas = StandardScaler()

maas_scale = sc_maas.fit_transform(maas)

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")

svr_reg.fit(egitim_scale,maas_scale)

predict = svr_reg.predict(egitim_scale)

plt.scatter(egitim_scale, maas_scale, color="red")
plt.plot(egitim_scale,predict,color="blue")
plt.show()







































