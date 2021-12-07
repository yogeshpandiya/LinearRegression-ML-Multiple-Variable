import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

a= np.array([1,2,3,4,5,2,3,4,6])
b= np.array([1000,1200,1300,1400,1500,1250,1390,1400,3600])
c= np.array([30,26,29,45,10,20,30,14,16])
d= np.array([10000,22000,33000,44000,55000,22500,43900,44000,66000])

data = {'BHK':a,
       'Area':b,
       'Age':c,
       'price':d}

df=pd.DataFrame(data)

reg=linear_model.LinearRegression()
reg.fit(df[['BHK','Area','Age']].values,df.price.values)

A=reg.predict([[2,1100,12]])
print(A)
