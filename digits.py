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
train_Y = data["label"]
train_Y = np.reshape(train_Y,(train_Y.shape[0],1))
train_X = np.transpose(train_X)
train_Y = np.transpose(train_Y)
train_X = train_X.as_matrix()

# initialize weights and betas
parameters = {}
layer_dims = [train_X.shape[0],8,10]
parameters = initialize_parameters_deep(layer_dims)


#forward propagation (fixed 2 layers)
Z1,cache = linear_forward(train_X,parameters["W1"],parameters["b1"])
A1,cache = leaky_relu(Z1)
Z2,cache = linear_forward(A1,parameters["W2"],parameters["b2"])
A2,cache = softmax(Z2)