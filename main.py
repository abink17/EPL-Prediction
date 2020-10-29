import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


epl_data = pd.read_csv("E:\\Projects\\EPL\\epl2020.csv")
print(epl_data.head())
print(epl_data.tail())

print(epl_data.describe())

print('COLUMN LIST:\n')
for i in epl_data.columns:
    print(i)

a = epl_data
print(a.iloc[:,0:1])
a.drop(a.iloc[:,0:1], inplace=True, axis=1)

print(a.columns.__len__())

x = epl_data.iloc[:,0:44]
print(x.columns)
print(x.head())

y = epl_data.iloc[:,7:8]
print(y.head())

z = epl_data.iloc[:,8:9]
print(z.head())

model_scored = ExtraTreesClassifier()
model_scored.fit(x,y)
print(model_scored.feature_importances_)
