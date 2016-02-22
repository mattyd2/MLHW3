import os
import numpy as np
import pickle
import random
from sklearn.cross_validation import train_test_split
import collections
from util import *
import timeit

def pegasos(X_train, y_train):
    num_iter = 1
    t = 2
    w = dict()
    lambda_reg = 0.75
    for epoch in range(num_iter):
        startTime = timeit.default_timer()
        for i in range(len(X_train)):
            t += 1
            stepSize = 1/(t*lambda_reg)
            # calculate the dot product of the w dictionary and the ith entry of the X_train which is a dictionary.
            # this returns a number
            dotWandX = dotProduct(w, X_train[i])
            # if the ith entry of the y_train dictionary, 1 or -1, multiplied by the dotWandX is less than one then update the
            # gradient of W by calculating the gradient using stepSize (number), y[i] (number), and X[i] dictionary.
            if y_train[i]*dotWandX < 1:
                # take the weights dict, "w", and update each entry by multipling the value by (1-stepSize*lambda_reg)
                increment(w,-stepSize*lambda_reg,w)
                # take the weights dict, "w", and update each entry by multipling the value by (stepSize*y_train[i]) value and each value of the X_train[i] dict.
                increment(w,stepSize*y_train[i],X_train[i])
            else:
                # if the value of y[i] multiplied by dotWandX is less than 1, then just update w by taking it's current value and multiplying it by (1-stepSize*Lambda_reg)
                increment(w,-stepSize*lambda_reg,w)
        endTime = timeit.default_timer()
        print "Run Time", endTime - startTime
    return w

def pegasosv2(X_train, y_train):
    num_iter = 1
    t = 2
    w = dict()
    s = 2
    lambda_reg = 0.75
    for epoch in range(num_iter):
        startTime = timeit.default_timer()
        for i in range(len(X_train)):
            t += 1
            # calculate the step size
            stepSize = 1/(lambda_reg*t)
            # calculate the dot product of the w dictionary and the ith entry of the X_train which is a dictionary.
            # this returns a number
            dotWandX = dotProduct(w, X_train[i])
            # calculate the scalar value of s so we can use it to scale w
            s *= (-lambda_reg*stepSize)
            # if the ith entry of the y_train dictionary, 1 or -1, multiplied by the dotWandX is less than one then update the
            # gradient of W by calculating the gradient using stepSize (number), y[i] (number), and X[i] dictionary.
            if y_train[i]*dotWandX < 1:
                # take the weights dict, "w", and update each entry by multipling the value by (stepSize*y_train[i]) value and each value of the X_train[i] dict.
                increment(w,stepSize*y_train[i]/s,X_train[i])
            else:
                # if the value of y[i] multiplied by dotWandX is less than 1, then just update w by taking it's current value and multiplying it by (1-stepSize*Lambda_reg)
                increment(w,-stepSize*lambda_reg,w)
        endTime = timeit.default_timer()
        print "Run Time", endTime - startTime
    return w
