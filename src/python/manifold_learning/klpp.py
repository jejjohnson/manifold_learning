from __future__ import division
from __future__ import absolute_import

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, spdiags, identity

from utils.graph_construction import create_laplacian, create_adjacency, \
                                     create_feature_mat, maximum, \
                                     compute_adjacency

from utils.eigenvalue_decomposition import EigSolver


class KernelLocalityPreservingProjections(BaseEstimator, TransformerMixin):
    """ Scikit-Learn compatible class for Kernel Locality Preserving
    Projections.
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
                 random_state = 0,
                 # kernel matrix parameters
                 kernel = "linear",
                 degree_kernel = 3,
                 gamma_kernel = None,
                 coef0_kernel = 1,
                 n_jobs_kernel = 1):
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
        self.sparse = sparse
        self.kernel = kernel
        self.degree_kernel = degree_kernel
        self.gamma_kernel = gamma_kernel
        self.coef0_kernel = coef0_kernel
        self.n_jobs_kernel = n_jobs_kernel

    def fit(self, X, y=None):

        # TODO: handle sparse case of data entry
        # check the array
        X = check_array(X)

        W = compute_adjacency(X,
                              n_neighbors=self.n_neighbors,
                              weight=self.weight,
                              affinity=self.affinity,
                              metric=self.metric,
                              neighbors_algorithm=self.neighbors_algorithm,
                              gamma=self.gamma,
                              trees=self.trees,
                              n_jobs=self.n_jobs)
        print('Computing Kernel Matrix...')
        K = self._get_kernel(X)
        K_model = KernelCenterer()
        K_model.fit(K)
        K = K_model.transform(K)
        print('Done!')

        # compute the projections into the new space
        _, self.projection_ = self._spectral_embedding(K, W)

        return self

    def transform(self, X, y=None):
        """Transform X.
        """

        # check the array X
        X = check_array(X)

        # get kernel matrix

        print('Projecting Data...')
        K = self._get_kernel(X, y)
        K_model = KernelCenterer()
        K_model.fit(K)
        K = K_model.transform(K)
        print('Done!')

        return K.dot(self.projection_)


    def _get_kernel(self, X, y=None):
        params = {"gamma": self.gamma_kernel,
                  "degree": self.degree_kernel,
                  "coef0": self.coef0_kernel}
        return pairwise_kernels(X, Y=y, metric=self.kernel,
                                n_jobs=self.n_jobs_kernel,
                                filter_params=True, **params)

    def _spectral_embedding(self, K, W):

        # create the laplacian and diagonal degree matrix
        self.L, self.D = create_laplacian(W, norm_lap=self.norm_lap,
                                          method='sklearn')

        # tune the generalized eigenvalue problem with necessary
        # parameters
        print('Tuning Eigenvalue Problem...')
        A, B = self._embedding_tuner(K)
        print('Done!')

        # Fit the generalized eigenvalue problem object to the
        # parameters

        print('Solving Eigenvalue Problem...')
        eig_model = EigSolver(n_components=self.n_components,
                              eig_solver=self.eig_solver,
                              sparse=True,
                              tol=self.tol,
                              norm_laplace=self.norm_lap)

        # return the eigenvalues and eigenvectors
        return eig_model.find_eig(A=A, B=B)

    def _embedding_tuner(self, K):

        # choose which normalization parameter to use
        if self.normalization == 'identiy':

            B = identity(n=np.shape(self.L)[0], format='csr')

        else:
            B = self.D

        # create feature matrices
        A = K.dot(self.L.dot(K))
        B = K.dot(B.dot(K))


        # get the sparsity cases
        return A, B
