from __future__ import division
from __future__ import absolute_import

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, spdiags, identity

from utils.graph_construction import create_laplacian, create_adjacency, \
                                     create_feature_mat, maximum, \
                                     compute_adjacency

from utils.eigenvalue_decomposition import EigSolver


class LocalityPreservingProjections(BaseEstimator, TransformerMixin):
    """ Scikit-Learn compatible class for Schroedinger Eigenmaps
    TODO: Parameters and Returns
    """
    def __init__(self,
                 # eigenvalue solver initials
                 n_components=2,
                 eig_solver = 'dense',
                 norm_lap = False,
                 tol = 1E-12,
                 # eigenvalue tuner initials
                 normalization = None,
                 # knn problem initials
                 n_neighbors = 2,
                 neighbors_algorithm = 'brute',
                 metric = 'euclidean',
                 n_jobs = 1,
                 weight = 'heat',
                 affinity = None,
                 gamma = 1.0,
                 trees = 10,
                 # general problem parameters
                 sparse = False,
                 random_state = 0):
        self.n_components = n_components
        self.eig_solver = eig_solver
        self.norm_lap = norm_lap
        self.tol = tol
        self.normalization = normalization
        self.n_neighbors = n_neighbors
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        self.weight = weight
        self.affinity = affinity
        self.gamma = gamma
        self.trees = trees
        self.sparse = sparse,
        self.random_state = random_state

    def fit(self, X, y=None):

        # TODO: handle sparse case of data entry
        # check the array
        X = check_array(X)

        # compute the adjacency matrix for X
        W = compute_adjacency(X,
                              n_neighbors=self.n_neighbors,
                              weight=self.weight,
                              affinity=self.affinity,
                              metric=self.metric,
                              neighbors_algorithm=self.neighbors_algorithm,
                              gamma=self.gamma,
                              trees=self.trees,
                              n_jobs=self.n_jobs)

        # compute the projections into the new space
        _, self.projection_ = self._spectral_embedding(X, W)

        return self

    def transform(self, X):

        # check the array and see if it satisfies the requirements
        X = check_array(X)
        if self.sparse:
            return X.dot(self.projection_)
        else:
            return np.dot(X, self.projection_)

    def _spectral_embedding(self, X, W):

        # create the laplacian and diagonal degree matrix
        self.L, self.D = create_laplacian(W, norm_lap=self.norm_lap,
                                          method='sklearn')

        # tune the generalized eigenvalue problem with necessary parameters
        A, B = self._embedding_tuner(X)

        # Fit the generalized eigenvalue problem object to the parameters
        eig_model = EigSolver(n_components=self.n_components,
                              eig_solver=self.eig_solver,
                              sparse=self.sparse,
                              tol=self.tol,
                              norm_laplace=self.norm_lap)

        # return the eigenvalues and eigenvectors
        return eig_model.find_eig(A=A, B=B)

    def _embedding_tuner(self, X):

        # choose which normalization paramter to use
        if self.normalization == 'identity':

            B = identity(n=np.shape(self.L)[0], format='csr')

        else:
            B = self.D

        # create feature matrices
        A = create_feature_mat(X, self.L)
        B = create_feature_mat(X, B)

        # get the sparsity cases
        if not self.sparse:
            return A.toarray(), B.toarray()
        else:
            return A, B
