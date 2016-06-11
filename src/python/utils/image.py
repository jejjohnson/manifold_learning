# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:03:52 2015

@author: jemanjohnson
"""

import numpy as np
import matplotlib.pyplot as plt 
import os
import scipy.io 
from sklearn import preprocessing
from time import time
from sklearn.preprocessing import MinMaxScaler

# Image Reshape Function
def img_as_array(img, gt=False):
    """Takes a N*M*D image
    where:
        * N   - number of rows
        * M   - number of columns
        * D   - dimension of data
        
    Returns:
    --------
    Image as an array with dimensions - 
    
        (N*M) by D
    """
    if gt == False:
        img_array = img.reshape(
            img.shape[0]*img.shape[1], img.shape[2])
    else:
        img_array = img.reshape(
            img.shape[0]*img.shape[1])
        
    return img_array

# Image Normalization function
def standardize(data):
    """
    Quick function to standardize my data between 0 and 1
    """
    return MinMaxScaler().fit_transform(data)
    

# Define HSI X and y Ground Truth pairing function
def img_gt_idx(img, img_gt, printinfo=False):
    """Takes a flattened image array and 
    extracts the image indices that correspond
    to the ground truth that we have.
    """
    # Find the non-zero entries
    n_samples = (img_gt>0).sum()
    
    # Find the classification labels
    classlabels = np.unique(img_gt[img_gt>0])
    
    # Create X matrix containing the features
    X = img[img_gt>0,:]
    
    # Create y matrix containing the labels
    y = img_gt[img_gt>0]
    
    # Print out useful information
    if printinfo:
        
        print('We have {n} ground-truth samples.'.format(
            n=n_samples))
        print('The training data includes {n} classes: {classes}'.format(
            n=classlabels.size, classes=classlabels.T))
        print('Dimensions of matrix X: {sizeX}'.format(sizeX=X.shape))
        print('Dimensions of matrix y: {sizey}'.format(sizey=y.shape))
    
    return X, y
    
# 