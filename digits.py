# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from nn_functions import initialize_parameters_deep,linear_forward,leaky_relu,softmax

# data processing
data = pd.read_csv("train.csv")
train_X = data.drop(["label"],axis=1)
label_Y = data["label"]
label_Y = np.reshape(label_Y,(label_Y.shape[0],1))
label_Y = np.transpose(label_Y)
train_X = np.transpose(train_X)
train_X = train_X.as_matrix()
train_Y = np.zeros((10,label_Y.shape[1]))

for i in range(label_Y.shape[1]):
    train_Y[label_Y[0,i],i] = 1
    
# initialize weights and betas
parameters = {}
layer_dims = [train_X.shape[0],8,10]
parameters = initialize_parameters_deep(layer_dims)

#forward propagation (fixed 2 layers)
Z1,cache = linear_forward(train_X,parameters["W1"],parameters["b1"])
A1,cache = leaky_relu(Z1)
Z2,cache = linear_forward(A1,parameters["W2"],parameters["b2"])
A2,cache = softmax(Z2)

#compute cost
cost = -np.log(A2[label_Y[0,],np.arange(label_Y.shape[1])])
loss = np.mean(cost)

#backprop