# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:02:38 2016

@author: eman
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, f1_score, \
                            fbeta_score, recall_score
from skll.metrics import kappa
import numpy as np
import pandas as pd

# this is my classification experiment function which basically delegates
# which classification method I want to use on the data.
def classification_exp(Xtrain, Xtest, Ytrain, Ytest, model='LDA'):
    
    if model in ['LDA', 'lda']:
        return lda_pred(Xtrain, Xtest, Ytrain, Ytest)
    else:
        raise ValueError('Sorry, the {m} model not available for' \
            'classification. Please use LDA for the time'\
            'being.'.format(m=model))
        
# my simple naive LDA function to classify my data. Since I have 
# multiple datasets, I loop through each dataset in my list and
# and perform classificaiton on that
#---------
# LDA Prediction
#---------------    
def lda_pred(Xtrain, Xtest, Ytrain, Ytest):
    """ Simple Naive Implementation of the the LDA
    """
    # empty list for the predictions
    Ypred = []
    
    # loop through and perform classification
    for xtrain, xtest, ytrain, ytest in zip(Xtrain,Xtest,
                                            Ytrain, Ytest):
        # initialize the model                
        lda_model = LDA()
        
        # fit the model to the training data
        lda_model.fit(xtrain, ytrain.ravel())
        
        # save the results of the model predicting the testing data
        Ypred.append(lda_model.predict(xtest))
    
    # return this list    
    return Ypred    

    

def accuracy_stats(Ypred, Ytest):
    
    stats = {}
    
    statkeys = ['AA', 'AP', 'f1', 'recall', 'kappa']
    for key in statkeys:
        stats[key] = []
   

    for ypred, ytest in zip(Ypred, Ytest):
        
        stats['AA'].append(accuracy_score(ytest.ravel(), ypred.ravel()))
        stats['AP'].append(precision_score(ytest.ravel(), ypred.ravel()))
        stats['f1'].append(f1_score(ytest.ravel(), ypred.ravel()))
        stats['recall'].append(recall_score(ytest.ravel(), ypred.ravel()))
        stats['kappa'].append(kappa(ytest.ravel(), ypred.ravel()))
        
    return stats

# the same function as before except with list comprehension
# (trying to practice that pythonic-ism a bit)    
def accuracy_statsv2(Ypred, Ytest):
    stats = {}
    
    statkeys = ['AA', 'AP', 'f1', 'recall', 'kappa']
    for key in statkeys:
        stats[key] = []
   
    stats['AA'] = [accuracy_score(ytest.ravel(), ypred.ravel()) for \
                   ypred, ytest in zip(Ypred, Ytest)]
    


    

def exp_runs(trials = 2):
    
    stats = [None]*trials
    
    Ystats = {'AA':None, 'AP':None}
    
    
    for key in Ystats:
        Ystats[key] = stats
    
    trial_num = 0
    while trial_num < trials:
        
        Ystats['AA'][trial_num], Ystats['AP'][trial_num] = run_exp()
        
        trial_num += 1
    
    return results_avg(Ystats)
    

        
        