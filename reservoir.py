# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:24:10 2018

@author: Alex
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from scipy.special import logit,expit


'''
10000 time steps
input size 100
reservoir size 1000
output size 10
'''


t = np.linspace(0, 1, 10000)
train_length = 3000
predict_length = len(t) - train_length
n_inputs = 500
reservoir_size = 10000
runup = 100

u = np.zeros(len(t)+100)
u[0:30] = np.random.rand(30)
for i in range(31,len(t)+100):
    u[i] = u[i-1] + 0.3*u[i-30]/(0.425 + u[i-30]**10) - 0.1*u[i-1]


# u = np.sin(2*np.pi*10*t) + np.exp(np.sin(2*np.pi*50*t)) + 0.0001*np.random.randn(len(t))
u = (u[100:] - np.min(u[100:])+0.05)/(0.2+np.max(u[100:]) - np.min(u[100:]))
x = np.random.randn(reservoir_size)
W = 0.1*nx.adjacency_matrix(nx.fast_gnp_random_graph(reservoir_size, 5/reservoir_size)).multiply(np.random.rand(reservoir_size, reservoir_size))#nx.relaxed_caveman_graph(20,50,4/50/50)
W = W/max(abs(linalg.eigs(W)[0]))*0.5
W_in = 0.5*np.random.randn(reservoir_size, n_inputs)
# W_out = 0.01*np.random.randn(1, 1200)
W_fb = 0.001*np.random.rand(reservoir_size)

# train

x_mat = np.zeros([train_length, reservoir_size+n_inputs])
y_target = []
for i in range(train_length):
    x = np.tanh(W_in.dot(u[i:i + n_inputs]) + W.dot(x) + W_fb.dot(u[i + n_inputs]))
    x_mat[i, :] = np.concatenate((u[i:i + n_inputs], x))
    y_target.append(u[i+n_inputs])

first = logit(np.array(y_target[runup:])).T.dot(x_mat[runup:])
second = np.linalg.pinv(x_mat[runup:].T.dot(x_mat[runup:]))
W_out = first.dot(second)

inputs = u[train_length:train_length+n_inputs]
for j in range(predict_length):
    x = np.tanh(W_in.dot(inputs[j:j + n_inputs]) + W.dot(x) + W_fb.dot(inputs[-1]))
    inputs = np.append(inputs, expit(W_out.dot(np.concatenate((inputs[j:j + n_inputs], x)))))


plt.plot(range(train_length+n_inputs,len(t)),u[train_length+n_inputs:],'-.')
plt.plot(range(train_length+n_inputs,len(t)),inputs[n_inputs:-n_inputs])


plt.figure()
plt.plot(u[train_length+n_inputs:]-inputs[n_inputs:-n_inputs])
plt.show()