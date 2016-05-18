# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:22:07 2016

@author: eman
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import hdidx

# Find the k-nearest neighbours
def knn_scikit(data, n_neighbors=4, method='brute'):
    """
    My k-nn function for finding the k-nn using different methods
    from the sklearn library.

    Methods available:

    * brute
    * kd_tree
    * ball_tree
    """

    # Set the k-nearest neighbours to the value inputed +1
    n_neighbors += 1


    if method in ['brute', 'kd_tree', 'ball_tree']:

        # Define the learning function with appropriate parameters
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm=method)

        # Fit the learning function to the data
        nbrs.fit(data)

        # Extract the k-nearest neighbors
        distVal, idx = nbrs.kneighbors(data)

    return distVal, idx

# Find approximate nearest neighbors
def knn_annoy(data, metric='euclidean',
              n_neighbors=10,
              trees=10):
    """My Approximate Nearest Neighbors function (ANN)
    using the annoy package.
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    """
    datapoints = data.shape[0]
    dimension = data.shape[1]
    
    # initialize the annoy database
    ann = AnnoyIndex(dimension)
    
    # store the datapoints
    for (i, row) in enumerate(data):
        ann.add_item(i, row.tolist())
        
    # build the index
    ann.build(trees)
    
    # find the k-nearest neighbors for all points
    idx = np.zeros((datapoints, n_neighbors), dtype='int')
    distVals = idx.copy().astype(np.float)
    
    # extract the distance values
    for i in range(0, datapoints):
        idx[i,:] = ann.get_nns_by_item(i, n_neighbors)
        
        for j in range(0, n_neighbors):
            distVals[i,j] = ann.get_distance(i, idx[i,j])
            
    return distVals, idx

# Compute the weights for the distance matrix
def dist_weights(distVal, method='heat',
                 sigma=1):
    """
    Compute the weights for the distance values
    """

    if method in ['angle']:       # The angle weight

        distValWeight = np.exp(-np.arcose(1 - distVal))
        
        # check for nan's (this tends to happen sometimes)
        if np.isnan(distValWeight):
            raise ValueError('Sorry...there are nans.'
            'Please use the heat kernel for now.')

        
    elif method in ['heat']:      # The heat weight kernel
        distValWeight = np.exp( - np.divide( distVal**2, sigma**2 ) )
        
    else:
        raise ValueError('Error! Need a method for the distance value')

    return distValWeight