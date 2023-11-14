import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 3, init = "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
    
plt.plot(range(1,11),sonuclar)

#%% k-means exercise

import pandas as pd

home_data = pd.read_csv("housing.csv",usecols=["longitude", "latitude","median_house_value"])
home_data.head()

import seaborn as sns

sns.scatterplot(data=home_data, x="longitude",y="latitude",hue="median_house_value")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data[["latitude","longitude"]], home_data[["median_house_value"]],test_size=0.33,random_state=0)


from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state= 0 , init = "k-means++")
kmeans.fit(X_train_norm)






















