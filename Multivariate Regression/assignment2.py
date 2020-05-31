# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:54:33 2020

@author: harsh
"""

#%%
             # MULTIVARIATE REGRESSION USING LEAST SQUARES-> 
import pandas as pd
import numpy as np
import math 
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

train_raw = pd.read_csv(r'Train (1).csv',low_memory=False)
test = pd.read_csv(r'Test.csv',low_memory= False)
train_raw.head()

def MULTIREG(x,y):
    """Multivariate coefficient determination
    parameters:
        x: pandasdataframe
        y: pandasdataframe
    returns: values of linear regression 
            coefficients in a numpy column 
            matrix.
    """
    A_val = pd.DataFrame(x)
    one =[]
    for i in range(len(x)):
        one.append(float(1))
    A_val['ones'] = one
    #preparing matrix A with column of 1's.
    A= A_val.values # numpy array of values of A
    #print(type(A))
    At = A.transpose() #transpose of A
    #print(type(At))
    AtA = np.matmul(At,A) # At*A 
    AtA_mat = np.matrix(AtA) # converting to matrix
    AtA_inv = (AtA_mat.I) # calculating inverse
    b = np.array([y.values]).T # transposing the B matrix
    Atb = np.matmul(At,b)
    Atb_mat = np.matrix(Atb) # converting to matrix
    Res = np.matmul(AtA_inv,Atb_mat)
    return Res
    
def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def predict(x,coeffs):
    A_val = pd.DataFrame(x)
    one =[]
    for i in range(len(x)):
        one.append(float(1))
    A_val['ones'] = one
    #preparing matrix A
    A= A_val.values 
    #numpy array of values of A
    res = np.matmul(A,coeffs)
    Res = res.tolist()
    return Res
    
features = []
for i in range(1,6):
    features.append("feature_"+str(i))
features
y = train_raw['target']
train = train_raw[features]
print(train)

x_train =train[:1300]
x_valid =train[1300:]
y_train =y[:1300]
y_valid =y[1300:]

#print(x_train.head(),"\n",y_train.head())

''' y_train represents my b vector.
 x_train is the matrix A.
 '''

model = MULTIREG(x_train,y_train)
#print(type(model))

preds_train= predict(x_train,model)
preds_valid= predict(x_valid,model)
pred_trn_list=[]
pred_valid_list=[]
for i in preds_train:
    pred_trn_list.append(i[0])
for i in preds_valid:
    pred_valid_list.append(i[0])
pred_valid_list[:10]
#%%
max(y_train)
#%%
print(rmse(pred_trn_list,y_train))
print(rmse(pred_valid_list,y_valid))
