# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:57:14 2016

@author: eman
"""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, spdiags, diags
from sklearn.utils.graph import graph_laplacian
from utils.nearestneighbor_solver import knn_scikit, knn_annoy
from utils.knn_solvers import KnnSolver

# compute the weighted adjacency matrix
def compute_adjacency(X, n_neighbors=5, affinity=None,weight='heat',
                      sparse=False, neighbors_algorithm='brute',
                      metric='euclidean', trees=10, gamma=1.0,
                      n_jobs=None):


     #-----------------------------------
     # K or Approximate Nearest Neighbors
     #-----------------------------------

     # initialize knn model with available parameters
     knn_model = KnnSolver(n_neighbors=n_neighbors,
                           nn_algorithm=neighbors_algorithm,
                           n_jobs=n_jobs,
                           affinity=affinity,
                           weight=weight,
                           gamma=gamma,
                           trees=trees,
                           metric=metric)

     # find the nearest neighbor indices and distances
     A_data, A_ind = knn_model.find_knn(X)

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

     return W


# Create Sparse Weighted Adjacency Matrix
def create_adjacency(distance_vals, indices):
    """This function will create a sparse symmetric weighted adjacency matrix
    from nearest neighbors and their corresponding distances.

    Parameters
    -----------
    distance_vals : numpy [MxN]
        an MxN array where M are the number of data points and N are the N-1
        nearest neighbor distances connected to that data point M.

    indices : array [MxN]
        an MxN array where M are the number of data points and N are the N-1
        nearest neighbors connected to that data point M.

    Returns
    --------
    Adjacency Matrix : array, sparse [MxM]
        a sparse MxM sparse weighted adjacency  matrix.
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
                     sparse=None):
    """Finds the Graph Laplacian from a Weighted Adjacency Matrix

    Parameters
    ----------
    * Adjacency       - a sparse NxN array

    Returns
    -------
    * Laplacian       - an NxN laplacian array
    * Diagonal        - an NxN diagonal array
    """
    L, D = graph_laplacian(Adjacency, normed=norm_lap,
                           return_diag=True)
    D = spdiags(data=D,
                diags=[0],
                m=Adjacency.shape[0],
                n=Adjacency.shape[0])
    return L, D



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
def laplacian_test():
    from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
    from sklearn.preprocessing import MinMaxScaler
    from scipy.sparse import csr_matrix, linalg
    import time as time
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')


    A = csr_matrix(make_spd_matrix(100))

    L, D, t = [], [], []
    for method in ['personal', 'sklearn']:
        t0 = time.time()
        temp_L, temp_D = create_laplacian(A)
        t1 = time.time()
        L.append(temp_L); D.append(temp_D)
        t.append(t1-t0)



    fig, ax = plt.subplots(nrows=1, ncols=2)



    ax[0].spy(L[0], precision=1E-10, markersize=.2)
    ax[0].set_title('My Method; {t:.2e} secs'.format(t=t[0]))
    ax[1].spy(L[1], precision=1E-10, markersize=.2)
    ax[1].set_title('Sklearn; {t:.2e} secs'.format(t=t[1]))
    plt.show()

    print(np.shape(L[0]), np.shape(L[1]))
    tol = 1E-1
    print('Different between the Laplacian Matrix' \
            'values close with tol: {tol}?'.format(tol=tol))

    assert (np.allclose(L[0].data, L[1].data, rtol=tol)), "False Laplacians not" \
    " the same."
    print('Test passed.')





if __name__ == "__main__":
    # sanity test
    laplacian_test()
