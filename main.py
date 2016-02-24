import os
import numpy as np
import collections
from pegasos import *
from load import *
import matplotlib.pyplot as plt

def lambdaListBuider(num_lambdas):
    lambda_reg_list = np.logspace(-4, 1, num_lambdas)
    return np.array(lambda_reg_list)

def compareprinter(weightsdictionary):
    for key in sorted(weightsdictionary)[:5]:
        print key, weightsdictionary[key]

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
    print "Run Time of Pegasos", endTime - startTime
    return classificationScore_list, w_list

def plotlambdascores(sigmun_score, lambda_reg_list, plottype):
    plt.plot(lambda_reg_list, sigmun_score, label="pegasos loss")
    plt.legend()
    plt.title(plottype)
    plt.xscale('log')
    plt.ylabel('loss')
    plt.xlabel('log(lambda value)')
    plt.savefig(plottype)
    plt.close()

def calcError(w_list, X, y):
    score_list = []
    for i in w_list:
        score_list.append(signumfunction(i, X, y))
    return score_list

def getbestLambda_Thetas(loss_hist_list, lambda_reg_list, theta_list):
    min_Loss = min(loss_hist_list)
    i = loss_hist_list.index(min_Loss)
    bestLambda = lambda_reg_list[i]
    bestThetas = theta_list[i]
    return bestLambda, bestThetas, min_Loss

def classificationAnalysis(X, y, w):
    incorrectlyclassified = []
    for i in range(len(y)):
        if y[i]*dotProduct(w, X[i])<0:
            incorrectlyclassified.append(X[i])
    return mdDotProduct(incorrectlyclassified[0], w), mdDotProduct(incorrectlyclassified[1], w)

def mdDotProduct(d1,d2):
    if len(d1) < len(d2):
        return mdDotProduct(d2,d1)
    return list((f,v,d1.get(f,0), abs(d1.get(f,0)*v)) for f,v in d2.items())

def runner(gram_size, num_lambdas, num_epochs):
    X_train, y_train, X_test, y_test = countTrainAndTest(gram_size)
    lambda_reg_list = lambdaListBuider(num_lambdas)
    # train_scores, w_list = testLambdaValues(X_train, y_train, lambda_reg_list, num_epochs, 'pegasos')
    train_scoresv2, wv2_list = testLambdaValues(X_train, y_train, lambda_reg_list, num_epochs, 'pegasosv2')
    # score_list, misclas_indicies_list = calcError(w_list, X_train, y_train)
    scorev2_list = calcError(wv2_list, X_train, y_train)
    # get the best lambda score
    bestLambda, bestwv2, min_Loss = getbestLambda_Thetas(scorev2_list, lambda_reg_list, wv2_list)
    # plotlambdascores(score_list, lambda_reg_list, 'pegasos')
    plotlambdascores(scorev2_list, lambda_reg_list, 'N_gram_Size - '+str(gram_size)+' - pegasosv2')
    # classificationAnalysis(w_list, misclas_indicies_list)
    misclassifiedList, misclassifiedList2 = classificationAnalysis(X_test, y_test, bestwv2)
    sorted_list = sorted(misclassifiedList2, key=lambda x: x[3], reverse=True)
    sorted_list2 = sorted(misclassifiedList, key=lambda x: x[3], reverse=True)
    print "\n ####### misclassified #1\n", sorted_list[:10]
    print "\n ####### misclassified #2\n", sorted_list2[:10]

def main():
    runner(1, 10, 2)
    runner(2, 10, 2)

if __name__ == "__main__":
    main()