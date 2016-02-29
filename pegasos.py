import numpy as np
import collections
from util import *
import timeit

# Question 6.2
def pegasos(X_train, y_train, lambda_reg, max_epochs):
    t = 2
    w = dict()
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        for i in range(len(X_train)):
            t += 1
            stepSize = 1/(t*lambda_reg)
            # if the ith entry of the y_train dictionary, 1 or -1, multiplied by the dotWandX is less than one then update the
            # gradient of W by calculating the gradient using stepSize (number), y[i] (number), and X[i] dictionary.
            if y_train[i]*dotProduct(w, X_train[i]) < 1:
                # take the weights dict, "w", and update each entry by multipling the value by (1-stepSize*lambda_reg)
                increment(w,-stepSize*lambda_reg,w)
                # take the weights dict, "w", and update each entry by multipling the value by (stepSize*y_train[i]) value and each value of the X_train[i] dict.
                increment(w,stepSize*y_train[i],X_train[i])
            else:
                # if the value of y[i] multiplied by dotWandX is less than 1, then just update w by taking it's current value and multiplying it by (1-stepSize*Lambda_reg)
                increment(w,-stepSize*lambda_reg,w)
    return w

# Question 6.3
def pegasosv2(X_train, y_train, lambda_reg, max_epochs):
    t = 2
    w = dict()
    s = 1
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        for i in range(len(X_train)):
            t += 1
            stepSize = 1/(t*lambda_reg)
            s *= (1-stepSize*lambda_reg)
            if s*y_train[i]*dotProduct(w, X_train[i]) < 1:
                increment(w,(stepSize*y_train[i])/s,X_train[i])
    increment(w,(s-1),w)
    return w
