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


#%% DECISION TREE


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color="red")
plt.plot(x, r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

# %% RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y)

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color="red")
plt.plot(x, rf_reg.predict(X),color="blue")

plt.plot(x,rf_reg.predict(Z),color="green")
plt.plot(x,rf_reg.predict(K),color="yellow")


















































































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






























