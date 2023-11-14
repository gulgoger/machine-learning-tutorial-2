
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('maaslar_yeni.csv',sep =";")

x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]

#numpy array(Dizi) dönüşümü
X = x.values
Y = y.values

print(veriler.corr())

#%% linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#%% P-VALUE

import statsmodels.api as sm
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
#%%polynomial linear regression


# 4.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#predicts


model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())
    

# %%
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

#%%SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)


print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),X)
print(model3.fit().summary())


#%% DECISION TREE REGRESSION


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print("DT OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())   

# %% RANDOM FOREST REGRESSION

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y)


print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


#%% R^2 -- R SQUARE
from sklearn.metrics import r2_score
print("Random Forest R2 degeri")
print(r2_score(Y,rf_reg.predict(X)))

print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))

# %% özet R2 değerleri
print("------------")
print("Linear R2 degeri")
print(r2_score(Y, lin_reg.predict(X)))

print("Polynomial R2 degeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print("SVR R2 degeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print("Decision Tree R2 degeri")
print(r2_score(Y, r_dt.predict(X)))  

print("Random Forest R2 degeri")
print(r2_score(Y, rf_reg.predict(X)))

































































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('maaslar.csv',sep =";")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#numpy array(Dizi) dönüşümü
X = x.values
Y = y.values


#%% linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.show()

#%%polynomial linear regression
# 2. dereceden polinom

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

# 4.dereceden polinom

poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color="blue")
plt.show()

#predicts
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


# %%
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)


plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color="blue")


print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))






























































































