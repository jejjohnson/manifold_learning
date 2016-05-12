# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:57:14 2016

@author: eman
"""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, spdiags, diags
from sklearn.utils.graph import graph_laplacian
from nearestneighbor_solver import knn_scikit, knn_annoy

# compute the weighted adjacency matrix
def compute_adjacency(X,
                    n_neighbors=5,
                    affinity=None,
                    weight='heat',
                    sparse=False,
                    neighbors_algorithm='brute',
                    metric='euclidean',
                    trees=10,
                    gamma=1.0):
     
     
     #-----------------------------------
     # K or Approximate Nearest Neighbors
     #-----------------------------------
     
     # the scikit learning libraries
     if neighbors_algorithm in ['brute', 'kd_tree', 'ball_tree']:
         
         A_data, A_ind = knn_scikit(X,
                                    n_neighbors=n_neighbors,
                                    method=neighbors_algorithm)
     
     # the spotify annoy ann library                               
     elif neighbors_algorithm == 'annoy':
         
         A_data, A_ind = knn_annoy(X, metric=metric,
                                   n_neighbors=n_neighbors,
                                   trees=trees)
                         
     else:
         raise ValueError('Sorry. Unrecognized knn algorithm.')
     
     #---------------------------------
     # Adjacency matrix and Data Kernel
     #---------------------------------
     
     # start constructing the adjacency matrix
     W = create_adjacency(A_data, A_ind)
     
     if weight == 'connectivity':
         raise ValueError('Sorry. Connectivity currently fails.')             
         W.data = 1
         
         
     elif weight == 'heat':
         W.data = np.exp(-W.data**2 / gamma**2)
        
         
     elif weight == 'angle':
         W.Data = np.exp(-np.arccos(1-W.data))
         
     else:
         raise ValueError('Sorry. Unrecognized affinity weight')

     if sparse:
         return W
     else:
         return W.todense() 

# Create Sparse Weighted Adjacency Matrix
def create_adjacency(distance_vals, indices):
    """This function will create a sparse symmetric
    weighted adjacency matrix from nearest neighbors
    and their corresponding distances.

    Parameters:
    -----------
    * idx                     - an MxN array where M are the number
                                of data points and N are the N-1
                                nearest neighbors connected to that
                                data point M.

    *  distance_vals          - an MxN array where M are the number
                                of data points and N are the N-1
                                nearest neighbor distances connected
                                to that data point M.

    Returns:
    --------
    Adjacency Matrix          - a sparse MxM sparse weighted adjacency
                                matrix.
    """
    # Separate, tile and ravel the neighbours from their
    # corresponding points
    row = np.tile( indices[:, 0].T, indices.shape[1]-1).T
    col = np.ravel( indices[:, 1:], order='F')
    data = np.ravel( distance_vals[:, 1:], order='F')

    # Create the sparse matrix
    W_sparse = csr_matrix( ( data, (row, col) ),
                          shape=(indices.shape[0],
                          indices.shape[0] ) )

    # Make sure the matrix is symmetric
    W_sparse_symmetric = maximum(W_sparse, W_sparse.T)
    
    return W_sparse_symmetric


# Find the maximum elements between two sparse matrices
def maximum(A,B):
    """This gives you the element-wise maximum between two sparse 
    matrices of size (nxn)
    
    Reference
    ---------
        http://goo.gl/k0Yfmk
    """
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

# Find the Laplacian Matrix from an Adjacency Matrix
def create_laplacian(Adjacency, 
                     norm_lap=None,
                     method=None,
                     sparse=None):
    """Finds the Graph Laplacian from a Weighted Adjacency Matrix

    Parameters
    ----------
    * Adjacency       - an NxN array

    Returns
    -------
    * Laplacian       - an NxN laplacian array
    * Diagonal        - an NxN diagonal array
    """

    if method in ['personal', 'Personal']:
            
        # D is equal to the sum of the rows of W
        try:
            D = np.squeeze(np.asarray(Adjacency.sum(axis=0)))
        except:
            
            D = np.diag(np.sum(Adjacency,axis=1))
    
    
        return diags(D, 0, (D.shape[0], D.shape[0]), format='csr') - \
                Adjacency, diags(D, 0, (D.shape[0], D.shape[0]), format='csr')
    elif method in ['scikit', 'sklearn']:
        
        if norm_lap:
            return_diag, D = None, None
            L, _ = graph_laplacian(Adjacency,
                                   normed=norm_lap,
                                   return_diag=return_diag)
            return L, D
        else:
            return_diag = True
            L, D = graph_laplacian(Adjacency,
                                   normed=norm_lap,
                                   return_diag=return_diag)
        
        if sparse:
            try:
                L = csr_matrix(L)
                D = diags(D, [0], shape=(L))                
                return L, D
            except:
                return L, diags(D, [0], shape=(L)) 
        else:
            try:
                L = np.asarray(L)
                D = np.diag(D)
                return L, D
            except:
                return L, np.diag(D)
            
        
    else:
        raise ValueError('Need a method of graph construction.')

                    
# create feature based matrix
def create_feature_mat(X,A, sparse=None):
    """This is the feature-based matrix of the form:
        X^T A X and X^T B X
    
    Parameters
    ----------
    X : (nxm) array
        This array has n_data points by m_features. Typically a data 
        matrix of n samples and m features.
    A : (nxn) array
        This array has n_data points by n_data points. This is typically
        constructed as an Laplacian matrix, adjacency matrix or diagonal
        degree matrix.
  
    
    Returns:
    --------
    
    X^TAX : (mxm) array
        This array has m_features by m_features.
    """
    if sparse:
        
        return X.T.dot(A.dot(X))
    
    else:
        try: 
            return np.dot(X.T, np.dot(A.toarray(),X))
            
        except:
            return np.dot(X.T, np.dot(A,X))