import os
import numpy as np
import collections
from pegasos import *
from load import *
from util import *
import matplotlib.pyplot as plt

def lambdaListBuider(num_lambdas):
    if num_lambdas == 1:
        return np.array([0.1])
    else:
        lambda_reg_list = np.logspace(-5, 1, num_lambdas)
        return np.array(lambda_reg_list)

# Question 6.6
def testLambdaValues(X_train, y_train, lambda_reg_list, num_epochs, algo):
    print '\n################', algo, '################\n'
    classificationScore_list, w_list = [], []
    startTime = timeit.default_timer()
    for i in lambda_reg_list:
        if algo == 'pegasos':
            w = pegasos(X_train, y_train, i, num_epochs)
        elif algo == 'pegasosv2':
            w = pegasosv2(X_train, y_train, i, num_epochs)
        w_list.append(w)
        classificationScore_list.append(signumfunction(w, X_train, y_train))
    endTime = timeit.default_timer()
    print "Run Time of "+algo, endTime - startTime
    return classificationScore_list, w_list

# Question 6.6
def plotlambdascores(sigmun_score, lambda_reg_list, plottype):
    plt.plot(lambda_reg_list, sigmun_score, label="pegasos loss")
    plt.legend()
    plt.title(plottype)
    plt.xscale('log')
    plt.ylabel('loss')
    plt.xlabel('log(lambda value)')
    plt.savefig(plottype)
    plt.close()

# Question 6.6
def getbestLambda_Thetas(loss_hist_list, lambda_reg_list, theta_list):
    min_Loss = min(loss_hist_list)
    i = loss_hist_list.index(min_Loss)
    bestLambda = lambda_reg_list[i]
    bestThetas = theta_list[i]
    return bestLambda, bestThetas, min_Loss

# Questin 6.5
def calcError(w_list, X, y):
    score_list = []
    for i in w_list:
        score_list.append(signumfunction(i, X, y))
    return score_list

# Questin 6.5 - continued
def signumfunction(w, X, y):
    count = 0
    for i in range(len(X)):
        if y[i]*dotProduct(w, X[i])<0:
            count += 1
    return count/(1. * len(X))

# Question 7.1
def classificationAnalysis(X, y, w):
    incorrectlyclassified = []
    for i in range(len(y)):
        # determine if the test instance was correctly or incorrectly classified
        if y[i]*dotProduct(w, X[i])<0:
            incorrectlyclassified.append(X[i])
    # take the dot product of the first two incorrectly classified test values to understand which weight values are driving the classification
    return mdDotProduct(incorrectlyclassified[0], w), mdDotProduct(incorrectlyclassified[1], w)

# Question 7.1
def mdDotProduct(d1,d2):
    if len(d1) < len(d2):
        return mdDotProduct(d2,d1)
    return list((f,v,d1.get(f,0), abs(d1.get(f,0)*v)) for f,v in d2.items())

# Question 6.4
def compareprinter(weightsdictionary):
    for key in sorted(weightsdictionary)[:5]:
        print key, weightsdictionary[key]

def runner(X_train, y_train, X_test, y_test, gram_size, num_lambdas, num_epochs, algo, question):
    lambda_reg_list = lambdaListBuider(num_lambdas)

    # takes the data, algo type, and the parameters of the algo
    train_scores, w_list = testLambdaValues(X_train, y_train, lambda_reg_list, num_epochs, algo)

    if question == "Question 8.1":
        # Questin 6.5 - calculate the percent erro when predicting y using sign((w.T)x)
        score_list = calcError(w_list, X_train, y_train)
        for i in range(len(score_list)):
            print "Lambda Value - ", lambda_reg_list[i], " the percent error was", score_list[i]

        # Question 6.6 - searching for the best Lambda regularization value
        bestLambda, bestw, min_Loss = getbestLambda_Thetas(score_list, lambda_reg_list, w_list)

        # plot the scores of the decision fucntion against the corresponding lamdba values
        plotlambdascores(score_list, lambda_reg_list, 'N_gram_Size - '+str(gram_size)+' - '+algo)

        # Questin 7.1
        misclassifiedList, misclassifiedList2 = classificationAnalysis(X_test, y_test, bestw)
        sorted_list = sorted(misclassifiedList2, key=lambda x: x[3], reverse=True)
        sorted_list2 = sorted(misclassifiedList, key=lambda x: x[3], reverse=True)
        print "\n ####### misclassified in example #1\n"
        for i in sorted_list[:10]:
            print i
        print "\n ####### misclassified in example #2\n"
        for i in sorted_list[:10]:
            print i

        return min_Loss

    elif question == "Question 6.4":
        return w_list

# Question 6.4
def comparePegasosVersions(gram_size):
    # get the data to be used for the comparison
    X_train, y_train, X_test, y_test = countTrainAndTest(gram_size)

    # execute the two versions of the Pegasos Algorithm
    w = runner(X_train, y_train, X_test, y_test, gram_size, 1, 2, 'pegasos', "Question 6.4")
    wv2 = runner(X_train, y_train, X_test, y_test, gram_size, 1, 2, 'pegasosv2', "Question 6.4")

    print "\n ####### Top 5 Pegasos keys and weight values #1\n"
    compareprinter(w[0])
    print "\n ####### Top 5 Pegasosv2 keys and weight values #1\n"
    compareprinter(wv2[0])

# Question 8.1
def compareUniGramAndBiGram(gram_size, num_lambdas, num_epochs, algo, question):
    X_train, y_train, X_test, y_test = countTrainAndTest(gram_size)
    min_Loss = runner(X_train, y_train, X_test, y_test, gram_size, num_lambdas, num_epochs, algo, question)
    return min_Loss

def main():
    # Question 6.4
    print "\n ######### Question 6.4 ############# \n"
    comparePegasosVersions(1)

    # Question 8.1
    print "\n ######### Question 8.1 ############# \n"
    print "\n ######### Question 8.1 UNIGRAMS ############# \n"
    min_Loss = compareUniGramAndBiGram(1, 10, 2, 'pegasosv2', "Question 8.1")
    print "\n ######### Question 8.1 BIGRAMS ############# \n"
    min_Loss_biGrams = compareUniGramAndBiGram(2, 10, 2, 'pegasosv2', "Question 8.1")

    # Question 8.1 - continued
    print "\n ######### Question 8.1 - continued ############# \n"
    print "min loss -", min_Loss
    print "min_Loss_biGrams -", min_Loss_biGrams
    p = 1.0-(min_Loss_biGrams/min_Loss)
    print "p -", p
    standError = np.sqrt(p*(1-p)/1500)
    print "standError -", standError
    print p - (1.64*standError) - (0.5/1500)
    print p + (1.64*standError) + (0.5/1500)


if __name__ == "__main__":
    main()
