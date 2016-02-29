import os
import numpy as np
import pickle
import random
from sklearn.cross_validation import train_test_split
import collections


'''
Note: No obligation to use this code, though you may if you like.  Skeleton code is just a hint for people who are not familiar with text processing in python.
It is not necessary to follow.
'''
def folder_list(path,label,n):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        if n > 1:
            r = n_gramBuilder(r, n)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(None, symbols).strip(), lines)
    words = filter(None, words)
    return words

# This is used for question
def n_gramBuilder(r, n):
    return zip(*[r[i:] for i in range(n)])

def shuffle_data(n):
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "/Users/matthewdunn/Dropbox/NYU/Spring2016/MachineLearning/HW/hw3-sentiment/data/pos"
    neg_path = "/Users/matthewdunn/Dropbox/NYU/Spring2016/MachineLearning/HW/hw3-sentiment/data/neg"

    pos_review = folder_list(pos_path,1,n)
    neg_review = folder_list(neg_path,-1,n)

    review = pos_review + neg_review
    random.shuffle(review)
    train = review[:150]
    test = review[1500:]
    return train, test

'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure.
*Check it out. https://wiki.python.org/moin/UsingPickle
'''

# Question 4.1
def countTrainAndTest(n):
    train, test = shuffle_data(n)
    X_train, y_train = counter(train)
    X_test, y_test = counter(test)
    return X_train, y_train, X_test, y_test

# Question 5.1
def counter(data):
    y = []
    X = []
    for i in range(len(data)):
        y.append(data[i][-1])
        X.append(collections.Counter(data[i][:-1]))
    return X, y
