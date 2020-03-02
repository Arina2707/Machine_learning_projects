# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:29:34 2020

@author: Arina27
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data=pd.read_csv('close_prices.csv')
x=data.loc[:,"AXP":]

#learning pca
pca=PCA(n_components=10, random_state=241)
pca.fit(x)

dispersion=pca.explained_variance_ratio_

sum=0
for i, n in enumerate(pca.explained_variance_ratio_):
    sum+=n
    if sum>=0.9:
        break
print(str(i+1))

#apply PCA to data

x0=pd.DataFrame(pca.transform(x))[0]
print(x0)#values of first component for all dates

#Pearson correlation

data_dd=pd.read_csv('djia_index.csv')
pearson_corr=np.corrcoef(x0, data_dd.loc[:,"^DJI"])
print(pearson_corr)

#the company with the highest weight in first component
weight=pca.components_[0]

print(x.columns[np.argmax(weight)])