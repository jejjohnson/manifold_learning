# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:12:31 2016

@author: eman
"""
from __future__ import division
from __future__ import absolute_import

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, spdiags, identity

from utils.nearestneighbor_solver import knn_scikit, knn_annoy
from utils.graph_construction import create_laplacian, create_adjacency, \
                               create_feature_mat, maximum, \
                               compute_adjacency
from utils.eigenvalue_decomposition import EigSolver

import pandas as pd



class SchroedingerEigenmaps(BaseEstimator, TransformerMixin):
    """ Scikit-learn compatible class for Schroedinger Eigenmaps

    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold

    n_neighbors : integer
        number of neighbors to consider for constructing the adjacency
        matrix

    neighbors_algorithm : string ['brute'|'kd_tree'|'ball_tree'|'ann']

    affinity : string ['connectivity' | 'heat' | 'cosine']
        weight function to use for the weighted adjacency matrix

    gamma : integer

    Reference
    ---------

    Schroedinger Eigenmaps for the Analysis of Bio-Medical Data

    Schroedinger Eigenmaps w/ Nondiagonal Potentials for Spatial-Spectral
    Clustering of Hyperspectral Imagery


    """
    def __init__(self,
                 # eigenvalue solver initials
                 n_components=2,
                 eig_solver = 'dense',
                 norm_lap = False,
                 tol = 1E-12,

                 # eigenvalue tuner initials
                 normalization = None,
                 alpha = 17.78,
                 beta = 1.0,
                 mu = 1.0,

                 # knn problem initials
                 n_neighbors = 2,
                 neighbors_algorithm = 'brute',
                 metric = 'euclidean',
                 n_jobs = 1,
                 weight = 'heat',
                 affinity = None,
                 gamma = 1.0,
                 trees = 10,

                 # potential matrix initials
                 potential = None,
                 sp_neighbors = 4,
                 sp_affinity = 'heat',
                 eta = 1.0,


                 # general problem parameters
                 sparse = False,
                 random_state=0):

        # eigenvalue solver initials
        self.n_components = n_components
        self.eig_solver = eig_solver
        self.norm_lap = norm_lap
        self.tol = tol

        # eigenvalue tuner initials
        self.normalization = normalization
        self.alpha = alpha
        self.beta = beta
        self.mu = mu


        # knn problem initials
        self.n_jobs = n_jobs
        self.affinity = affinity
        self.weight = weight
        self.gamma = gamma
        self.eta = eta
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.neighbors_algorithm = neighbors_algorithm

        # potential matrix initials
        self.potential = potential
        self.sp_neighbors = sp_neighbors
        self.sp_affinity = sp_affinity
        self.metric = metric
        self.trees = trees

        # non-input initial parameters
        self.V_d = None
        self.V_s = None

        # general problem parameters
        self.sparse = sparse
        self.random_state = random_state



    def fit(self, X, y=None, X_img=None):

         if not X_img:
             X_img = X

         ''' TODO: contain the potential matrix choices within the
           internal potential matrix function'''
         # check the array and see if it satisfies the requirements
         X = check_array(X)

         # compute the weighted adjacency matrix for X

         W = compute_adjacency(X,
                               n_neighbors=self.n_neighbors,
                               weight=self.weight,
                               affinity=self.affinity,
                               neighbors_algorithm=self.neighbors_algorithm)

         # compute potential matrix
         self.V_s = self._potential(X, y=y, X_img=X_img)



         # compute normalization potential matrix
         self.V_d = self._normalization(X, y=y, X_img=None)

         # compute the projection into the new space

         _, self.embedding_ = self._spectral_embedding(X, W)
         return self


    # Compute the projection of X into the new space
    def transform(self, X):

          # check the array and see if it satisfies the requirements
         X = check_array(X)

         return self.embedding_

    # Normalization parameter
    def _normalization(self, X=None, y=None, X_img=None):

        if self.normalization == 'dissimilarity':
             return self._potential(X=X, y=None,
                                    potential=potential)

        elif self.normalization not in ['degree', 'identity', 'sema']:
             return None



    # function that deciphers which potential matrix to use
    def _potential(self, X, y=None, X_img=None):



         # The classifical laplacian matrix case
         if not self.potential:
             return None

         # compute spatial-spectral potential matrix
         elif self.potential == 'ssse':

             # get spatial coordinates for dataset (specifically images)
             X_spatial = get_spatial_coordinates(X_img)

             # find the k_nearest neighbors indices
             _, V_ind = knn_scikit(X, n_neighbors=self.sp_neighbors,
                                   method='brute')

             # return the spatial-spectral potential matrix
             return ssse_potential(X, X_spatial, V_ind,
                                   weight=self.sp_affinity)

         # create the similarity potential matrix
         elif self.potential == 'similarity':
             raise ValueError('Sorry. This method is unavailable at'
             'the moment.')

         # create the dissimilarity potential matrix
         elif self.potential == 'dissimilarity':
             raise ValueError('Sorry. This method is unavailable at'
             'the momement.')

         # create the partial knowledge potential matrix
         elif self.potential == 'pkssse':
             raise ValueError('Sorry. This method is unavailable at'
             'the moment.')

         else:
             raise ValueError('Sorry. Unrecognized Potential matrix.')

    # eigenvalue problem tuner
    def _embedding_tuner(self):


         # Choose which normalization parameter to use
         if self.normalization == 'identity':

             B = identity(n=np.shape(self.L)[0], format='csr')

         elif self.normalization == 'sema':

             beta = get_alpha(alpha=self.beta, D=self.D, V=self.V_d)
             B = self.D + beta * self.V_d

         elif self.normalization == 'dissimilarity':

             B = self.V_d

         else:
             B = self.D

         # choose which Laplacian/Schroedinger Operator to use

         if self.potential in ['ssse', 'similarity', 'pkssse']:

             alpha = get_alpha(alpha=self.alpha, L=self.L, V=self.V_s)
             A = self.L + self.alpha * self.V_s

         else:
             A = self.L

         # sparse case
         if not self.sparse:
             try:
                 A = A.toarray()
             except:
                 pass
             try:
                 B = B.toarray()
             except:
                 pass

         # return the values of A and B

         return A, B


    # compute the spectral embedding
    def _spectral_embedding(self, X, W):

        # create the laplacian and diagonal degree matrix
        self.L, self.D = create_laplacian(W,
                                          norm_lap=self.norm_lap,
                                          method='personal')

        # tune the generalized eigenvalue problem with parameters
        A, B = self._embedding_tuner()

        # Fit the parameters to an eigenvalue solver method
        eig_model = EigSolver(n_components=self.n_components,
                              eig_solver=self.eig_solver,
                              sparse=self.sparse,
                              tol=self.tol,
                              norm_laplace=self.norm_lap)

        # return the eigenvalues and eigenvectors
        return eig_model.find_eig(A=A,B=B)



# Module for extracting the spatial features of the data
def get_spatial_coordinates(data):
    """
    This function extracts the spatial dimensions of the data.
    (e.g. an image)
    It excepts two types of data inputs: 2D data and 3D data.
    - For the 2D data this will simulate the x neighbors
    - For the 3D data, this will simulate the x-y neighbors

    Parameters
    ----------
    - data: a 2D or 3D dense numpy array

    Returns
    -------
    - data: a 2D dense array


    """
    # Get the dimensions of the original data
    try:        # Try the case where we have a 3D data set
        nrows = data.shape[0]
        ncols = data.shape[1]
        ndims = data.shape[2]


    except:     # Except the case where we have a 2D dataset
        nrows = data.shape[0]; ncols = 1

    # Create a meshgrid for the spatial locations
    xv, yv = np.meshgrid(np.arange(0, ncols, 1),
                        np.arange(0, nrows, 1),
                        sparse=False)

    # ravel the data using the FORTRAN order system
    # for the x and y coordinates
    xv = np.ravel(xv, order='F')
    yv = np.ravel(yv, order='F')

    return np.vstack((xv,yv)).T


# Construct the Schroedinger Spatial-Spectral Potential Matrix
def ssse_potential(data,
                   clusterdata,
                   indices,
                   weight='heat',
                   sp_weight='heat',
                   sigma=1.0,
                   eta=1.0):
    """Constructs the: Schroedinger Spatial-Spectral Cluster Potential

    Parameters
    ----------

    * data          - MxD data array where M is the number of
                      data points and D is the dimension of the
                      data.
    * clusterdata   - Mx2 spatial data array for the clustering
                      clustering points where M is the number of
                      data points.
    indices: (M, N) array_like
        an MxN array where M are the number of data points and N
        are the N-1 nearest neighbors connected to that data point M.

    weight: str ['heat'|'angle'] (optional)
        The weight parameter as the kernel for the spatial-spectral
        difference.

    weight: str ['heat'|'angle'] (optional)
        The weight parameter as the kernel for the spatial-spectral
        difference.

    sigma: float, optional
        The parameter for the heat kernel for the spatial values.
        Default: 1.0

    eta: float, optional
        The parameter for the heat kernel.
        Default: 1.0


    Returns
    -------
    * Potential Matrix     - a sparse MxM potential matrix


    References:
    -----------

    Original Author:
        Nathan D. Cahill

    Code:
        https://people.rit.edu/ndcsma/code.html
    Website:
        https://people.rit.edu/ndcsma/

    Papers:
    N. D. Cahill, W. Czaja, and D. W. Messinger, "Schroedinger Eigenmaps
    with Nondiagonal Potentials for Spatial-Spectral Clustering of
    Hyperspectral Imagery," Proc. SPIE Defense & Security: Algorithms and
    Technologies for Multispectral, Hyperspectral, and Ultraspectral
    Imagery XX, May 2014

    N. D. Cahill, W. Czaja, and D. W. Messinger, "Spatial-Spectral
    Schroedinger Eigenmaps for Dimensionality Reduction and Classification
    of Hyperspectral Imagery," submitted.

    """
    # Number of data points and number of cluster potentials
    N = data.shape[0]; K = indices.shape[1]-1

    # Compute the weights for the Data Vector EData
    x1 = np.repeat(
            np.transpose(
            data[:, :, np.newaxis], axes=[0, 2, 1]), K, axis=1)

    x2 = data[indices[:,1:]].reshape((N, K, data.shape[1]))

    if weight == 'heat':
        WE = np.exp( - np.sum ( ( x1-x2 )**2, axis=2 ) / sigma**2)

    elif weight == 'angle':
        WE = np.exp( - np.arccos(1-np.sum( (x1-x2), axis=2 ) ) )

    else:
        raise ValueError('Unrecognized SSSE Potential weight.')

    # Compute the weights for the Clustering Data Vector CData
    x1 = np.repeat(
            np.transpose(
            clusterdata[:, :, np.newaxis], axes=[0, 2, 1]), K, axis=1)

    x2 = clusterdata[indices[:,1:]].reshape(( N, K, clusterdata.shape[1] ))

    if weight == 'heat':
        WC = np.exp( - np.sum ( ( x1 - x2 )**2, axis=2 ) / eta**2)

    elif weight == 'angle':
        WC = np.exp( - np.arccos(1-np.sum( (x1-x2), axis=2 ) ) )

    else:
        raise ValueError('Unrecognized SSSE Potential weight.')

    # Create NonDiagonal Elements of Potential matrix, V
    Vrow = np.tile( indices[:, 0], K)
    Vcol = np.ravel( indices[:, 1:], order='F' )
    V_vals = -WE*WC
    Vdata = np.ravel( V_vals, order='F')

    # Create the symmetric Sparse Potential Matrix, V
    V_sparse = csr_matrix( (Vdata, (Vrow, Vcol) ),
                           shape=( N,N ))
    # Make Potential Matrix sparse
    V_sparse = maximum(V_sparse, V_sparse.T)

    # Compute the Diagonal Elements of Potential Matrix, V
    V_diags = spdiags( -V_sparse.sum(axis=1).T, 0, N, N)

    # Return the sparse Potential Matrix V
    return V_diags+V_sparse


# create similarity and dissimilarity potential matrices
def sim_potential(X, potential='sim',
                  norm_lap=None,
                  method='personal',
                  sparse_mat=None):
    """Creates similarity or dissimilarity potential matrix.

    Parameters:
    ----------
    X               - a k list of (n1+n2) x (m) labeled data matrices where
                      n1 is the number of labeled samples and n2 is the
                      number of unlabeled samples. We assume that the
                      labeled entries are positive natural numbers and the
                      unlabeled entries are zero.

    Returns
    -------
    Ws              - a sparse (n1+n2)*k x (n1+n2)*k adjacency matrix
                      showing the connectivity between the corresponding
                      similar labels between the k entries in the list.
    Wd              - a sparse (n1+n2)*k x (n1+n2)*k adjacency matrix
                      showing the connectivity between the corresponding
                      dissimilar labels between the k entries in the list.

    TODO: References
    ----------------

    Tuia et al. - Semisupervised Manifold Alignment of Multimodal Remote
        Sensing Images
        http://isp.uv.es/code/ssma.htm
    Wang - Heterogenous Domain Adaption using Manifold Alignment
        https://goo.gl/QaLTA4

    """
    # create sparse matrix from array X
    new = {}
    new['csr'] = sparse.csr_matrix((X.shape[0],X.shape[0]), dtype='int')
    row = []; col = []; datapoint = []

    i = 0
    for (Xrow, Xcol, Xdata) in zip(X.row, X.col, X.data):

        # find all the datapoints in the Y matrix
        # greater than 0
        if Xdata > 0:

            # copy that data point with its row and column entry
            new['csr'][Xrow,:] = Xdata

        i += 1
    # Copy the original matrix transposed
    new['csrt'] = new['csr'].T.copy()

    # convert the matrix to a coordinate matrix
    new['coo'] = new['csr'].tocoo()
    new['coot'] = new['csrt'].tocoo()


    col_labels = ['rows', 'cols', 'data']


    # join the dataframes
    united_data = pd.concat([pd.DataFrame(data=np.array((new['coo'].row,
                                                         new['coo'].col,
                                                         new['coo'].data)).T,
                                          columns=col_labels),
                             pd.DataFrame(data=np.array((new['coot'].row,
                                                         new['coot'].col,
                                                         new['coot'].data)).T,
                                          columns=col_labels)])

    # group the data by the whole row to find the duplicates
    united_data_grouped = united_data.groupby(list(united_data.columns))

    # detect the row indices with similar values
    same_data_idx = [x[0] \
        for x in united_data_grouped.indices.values() if len(x)!=1]

    # extract the unique values
    same_data = united_data.iloc[same_data_idx]

    # create empty sparse matrix
    Ws = sparse.csr_matrix((X.shape[0],X.shape[0]), dtype='int')

    # set the unique value indcies to 1
    Ws[same_data['rows'], same_data['cols']] = 1

    if potential in ['sim']:
        return create_laplacian(Ws,
                                norm_lap=norm_lap,
                                method=method,
                                sparse=sparse_mat)

    # group the data by the two columns (row, col)
    united_cols_grouped = united_data.groupby(['rows','cols'])

    # detect the row indices with similar values for the two columns compared with data
    diff_cols_idx = [x[0] for x, z in zip(united_cols_grouped.indices.values(),
                                          united_cols_grouped['data'].indices.values())
                                          if (len(x)!=1 & len(z)!=1)]

    # extract those same values
    diff_data = united_data.iloc[diff_cols_idx]


    # convert back to csr format
    Wd = sparse.csr_matrix((X.shape[0],X.shape[0]), dtype='int')
    Wd[diff_data['rows'], diff_data['cols']] = 1

    return create_laplacian(Wd-Ws,
                                norm_lap=norm_lap,
                                method=method,
                                sparse=sparse_mat)


# Determine appropriate trade off parameter between L and D
def get_alpha(alpha, L, V):
    """Gives the suggested value of alpha:

    Trace of Potential
    ------------------ = Suggested Alpha
    Trace of Laplacian

    Parameters
    ----------
    * weight     - trade-off parameter between
                   the Laplacian matrix and the
                   Schroedinger Potential matrix
    * Laplacian  - Sparse Matrix Laplacian
    * Potential  - Sparse Potential Matrix

    Returns
    -------

    * Scaled Weight parameter to trade off the Laplacian
    and the Potential Matrix

    """
    return alpha * ( np.trace(L.todense()) / np.trace(V.todense()))
