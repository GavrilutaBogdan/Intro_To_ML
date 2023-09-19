#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:04:13 2023

@author: piripuz
"""

import pandas as pd
import numpy as np
import scipy.linalg as linalg
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm, scatter)
from mpl_toolkits.mplot3d import Axes3D
import array_to_latex as a2l
import seaborn as sns
import math

dict_r = {
        'B' : 0,
        'M' : 1
        }
df = pd.read_csv("./Prostate_Cancer.csv", index_col='id')
df.replace(dict_r, inplace=True)
X=df.drop('diagnosis_result', axis=1).to_numpy()
N,M = X.shape

Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
U,S,V = linalg.svd(Xc,full_matrices=False)
V = V.T
rho = (S*S) / (S*S).sum() 
Z = Xc @ V

figure()
plot(rho,'o-')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained value')


X_plot = np.linspace(50, 160, num=500)
Y_plot = np.vectorize((lambda x: x**2/(4*math.pi)))(X_plot)


sns.relplot(x=Z[:,0], y=Z[:,1], hue=df['diagnosis_result'])
corr = sns.relplot(x=X[:,2],y=X[:,3], hue=df['diagnosis_result'])
plot(X_plot, Y_plot, color='r')

sns.pairplot(df, hue='diagnosis_result')

'''three_d = plot()
plot_axes = three_d.axes(projection='3d')
plot_axes.scatter3D(Z[:,0],Z[:,1],Z[:,2])'''
#fig=figure()
#ax = Axes3D(fig, auto_add_to_figure=False)
#fig.add_axes(ax)
# sc = ax.scatter(Z[:,0], Z[:, 1], Z[:, 2], c=df['diagnosis_result'])