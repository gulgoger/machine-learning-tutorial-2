from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% SIMPLE LINEAR REGRESSION EXERCISE

df = pd.read_csv("USA_Housing.csv")
df.head()

df.info()

df.describe()

df.columns

sns.pairplot(df)

sns.distplot(df["Price"])

sns.heatmap(df.corr(), annot=True)

df.columns

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)


lm.coef_

X_train.columns


# %% MULTIPLE LINEAR REGRESSION

house = pd.read_csv("Boston.csv")
house.head()

house.info()
house.describe()

y = house["MEDV"]
X = house.drop(["MEDV"],axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=101)

X_train.shape, X_test.shape , y_train.shape, y_test.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)
model.intercept_
model.coef_

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
mean_absolute_error(y_test, y_pred)

mean_absolute_percentage_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)


#%% POLYNOMIAL LINEAR REGRESSION

dataset = pd.read_csv("position_salaries.csv")
X = dataset.iloc[:,1:2].values
y=  dataset.iloc[:,2].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

def viz_linear():
    plt.scatter(X,y, color="red")
    plt.plot(X, lin_reg.predict(X),color= "blue")
    plt.title("Truth or Bluff(Linear Regression)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()
    return

viz_linear()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly,y)


def viz_polynomial():
    plt.scatter(X,y, color="red")
    plt.plot(X, poly_reg.predict(X),color= "blue")
    plt.title("Truth or Bluff(Linear Regression)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()
    return
viz_polynomial()