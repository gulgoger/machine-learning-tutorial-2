import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('odev_tenis.csv')

Yas = veriler.iloc[:,1:4].values

# %%  categorical values -- kategorik verileri sayısal değerlere dönüştürme-- encoding

from sklearn import preprocessing
veriler2= veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
# one hot encoding (ohe) -- colomn başlıklarını etikete taşıyıp 1 veya 0 olarak etiketlemek


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
c = ohe.fit_transform(c).toarray()
print(c)


havadurumu = pd.DataFrame(data=c, index = range(14), columns=["o","r","s"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([sonveriler, veriler2.iloc[:,-2:]],axis=1)




# %% data train - test split

from sklearn.model_selection import train_test_split
    
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,:-1], test_size=0.33,random_state=0)


# %% multiple linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


#%% backward elimination

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog= sonveriler.ilog[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

sonveriler= sonveriler.iloc[:,1:]

X = np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog= sonveriler.ilog[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())


x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]


regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)















































