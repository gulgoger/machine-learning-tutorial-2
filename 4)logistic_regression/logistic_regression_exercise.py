import pandas as pd

data = pd.read_csv("train.csv")

x = data.iloc[:,5:6].values

y = data.iloc[:,4:5]

from sklearn.model_selection import train_test_split
  
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.25, random_state=0)



from sklearn.preprocessing import StandardScaler
  
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
  
print (xtrain[0:10, :])





