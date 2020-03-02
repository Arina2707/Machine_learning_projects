import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from math import exp
from scipy.spatial import distance

data=pd.read_csv('data-logistic.csv',header=None)

X=data[[1,2]].to_numpy()
Y_old=data[[0]].to_numpy()
Y_old=np.reshape(Y_old,np.size(Y_old))
X=np.asmatrix(X)
Y=np.asmatrix(Y_old)


#without regularization C=0
w1=np.array(0)
w2=np.array(0)
k=0.1

for i in range(10000):
    w1_prev, w2_prev = w1, w2
    w1=w1+k*np.mean(Y*X[:,0]*(1-(1./(1+exp(-1*Y*(w1*X[:,0]+w2*X[:,1]))))))
    w2=w2+k*np.mean(Y*X[:,1]*(1-(1./(1+exp(-1*Y*(w1*X[:,0]+w2*X[:,1]))))))
    if np.sqrt((w1_prev - w1) ** 2 + (w2_prev - w2) ** 2)<=1e-5:
       break

predictions1=[]

for j in range(len(X)):   
    t1=-w1*X[j,0]-w2*X[j,1]
    s=1/(1+exp(t1))
    predictions1.append(s)
quality_metric1=roc_auc_score(Y_old.tolist(),predictions1)
print(quality_metric1)

#with regularization C=10
