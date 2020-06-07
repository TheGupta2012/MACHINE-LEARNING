# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:29 2020

@author: harsh
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import statistics as st
def line(X,Y):
    mb=[0,0]
    x1= float(sum(X))
    y1= float(sum(Y))
    xy=0
    x2=0
    for i in range(len(X)):
        xy+=X[i]*Y[i]
        x2+=X[i]**2
    mb[1]=float((y1*x2-x1*xy)/(len(X)*x2-x1**2))
    mb[0]=float((xy/x2)-(x1*mb[1])/x2)
    return mb
df_x= pd.read_csv(r'Linear_X_Train.csv')
df_y= pd.read_csv(r'Linear_Y_Train.csv')
X= list(df_x.x)
Y= list(df_y.y)
mb=line(X,Y)
xval=[]
yval=[]
i=-4.00
while(i<6):
    xval.append(i)
    yval.append(mb[0]*i+mb[1])
    i+=0.001
for i in range(len(X)):
    plot.scatter(X[i],Y[i],marker='.')
plot.title("LINEAR REGRESSION WITH TRAINING SET - VISUALISED",fontsize=13,fontweight='bold')
plot.grid()
plot.text(-4,400,"Line is: Y= %f*X + %f" %(mb[0],mb[1]),fontsize=14.5)
plot.xlabel("Time Devoted By Student")
plot.ylabel("Performance achieved by Student")
plot.plot(xval,yval,color='magenta',linestyle='dotted',linewidth=5)
plot.savefig("Training",dpi=220)
plot.show()
X_TEST= pd.read_csv(r'Linear_X_Test.csv')
XT = list(X_TEST.x)
YT=[]
for k in XT:
    YT.append(mb[0]*k+mb[1])
plot.scatter(XT,YT,marker=".")
plot.title("TEST DATA SET GRAPH",color='green',fontsize=15,fontweight='bold')
plot.xlabel("Time Devoted By Student")
plot.ylabel("Performance achieved by Student")
plot.plot(xval,yval,c='r',linestyle="--")
plot.text(-4,450,"--",color='red',fontsize=15)
plot.text(-3.4,450,"PREDICTED LINE",fontsize=15,fontweight='bold')
Yfinal=pd.DataFrame(YT,columns=['Y_test'])
plot.savefig("TestGraph",dpi=220)
Yfinal.to_csv('Y_test_Harshit.csv')
Yfinal.head()
