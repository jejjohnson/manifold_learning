from __future__ import division
from __future__ import absolute_import

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, spdiags, identity

from utils.graph import create_laplacian, create_adjacency, \
                                     create_feature_mat, maximum, \
                                     compute_adjacency

from utils.eigenvalue_decomposition import EigSolver


class LaplacianEigenmaps(BaseEstimator):
    """ Scikit-Learn compatible class for Locality Preserving Projections

    Parameters
    ----------

    n_components : integer, optional, default=2
        number of features for the manifold (=< features of data)

    eig_solver : string ['dense', 'multi', 'sparse'], optional, default='dense'
        eigenvalue solver method

    norm_lap : bool, optional, default=False
        normalized laplacian or not

    tol : float, optional, default=1E-12
        stopping criterion for eigenvalue decomposition of the Laplacian matrix
        when using arpack or multi

    normalization : string ['degree', 'identity'], default = None ('degree')
        normalization parameter for eigenvalue problem

    n_neighbors :

    Attributes
    ----------

    _spectral_embedding :

    _embedding_tuner :


    References
    ----------

    Original Paper:
        http://www.cad.zju.edu.cn/home/xiaofeihe/LPP.html
    Inspired by Dr. Jake Vanderplas' Implementation:
        https://github.com/jakevdp/lpproj

    """
    def __init__(self, n_components=2, eig_solver = 'dense', norm_laplace = False,
                 eigen_tol = 1E-12, regularizer = None,
                 normalization = None, n_neighbors = 2,neighbors_algorithm = 'brute',
                 metric = 'euclidean',n_jobs = 1,weight = 'heat',affinity = None,
                 gamma = 1.0,trees = 10,sparse = True,random_state = 0):
        self.n_components = n_components
        self.eig_solver = eig_solver
        self.regularizer = regularizer
        self.norm_laplace = norm_laplace
        self.eigen_tol = eigen_tol
        self.normalization = normalization
        self.n_neighbors = n_neighbors
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        self.weight = weight
        self.affinity = affinity
        self.gamma = gamma
        self.trees = trees
        self.sparse = False,
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
        self.eigVals, self.embedding_ = self._spectral_embedding(X, W)

        return self

    # Compute the projection of X into the new space
    def fit_transform(self, X):
        # check the array and see if it satisfies the requirements
        X = check_array(X)
        self.fit(X)

        return self.embedding_


    def _spectral_embedding(self, X, W):

        # find the eigenvalues and eigenvectors
        return graph_embedding(adjacency=W, norm_laplace=self.norm_laplace,
                               normalization=self.normalization,
                               eig_solver=self.eig_solver,
                               eig_tol=self.eigen_tol)


def graph_embedding(adjacency,
                    norm_laplace = None,
                    norm_method = 'degree', normalization= None, mu=1.0,
                    ss_potential=None, alpha=17.78,
                    pl_potential=None, beta=1.0,
                    n_components=2,eig_solver=None,eig_tol=1E-12,):
    """
    Returns
    -------
    eigenvalues
    eigenvectors
    TODO - time elapse
    """
    # create laplacian and diagonal degree matrix
    L, D = create_laplacian(adjacency, norm_lap=norm_laplace)

    #-------------------------------
    # Tune the Eigenvalue Problem
    #-------------------------------
    # choose which normalization parameter to use
    if norm_method in ['degree', 'Degree', None]:   # standard laplacian
        B = D

    elif normalization in ['identity']:     # analogous to ratio cute
        B = identity(n=np.shape(L)[0], format='csr')

    else:
        raise ValueError('Not a valid normalization parameter...')

    # choose the regularizer
    if not ss_potential == None:            # spatial-spectral potential
        if not alpha:
            alpha = 17.78
        alpha = get_alpha(alpha, L, ss_potential)
        A = L + alpha * ss_potential

    elif not pl_potential == None:          # partial-labels potential
        beta = get_alpha(beta, L, pl_potential)
        A = L + beta * pl_potential

    else:                       # no potential (standard Laplacian)
        A = L

    #-------------------------------
    # Solve the Eigenvalue Problem
    #-------------------------------

    # initialize the EigSolver class
    eig_model = EigSolver(n_components=n_components,
                          eig_solver=eig_solver,
                          sparse=sparse,
                          tol=eig_tol,
                          norm_laplace=norm_laplace)

    # return the eigenvalues and eigenvectors
    return eig_model.find_eig(A=A, B=B)

def swiss_roll_test():

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from time import time

    from sklearn import manifold, datasets
    from sklearn.manifold import SpectralEmbedding

    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    n_neighbors=10
    n_components=2

    # original scikit-learn lE algorithm
    t0 = time()
    ml_model = SpectralEmbedding(affinity='nearest_neighbors',
                                 n_neighbors=n_neighbors,
                                 n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='scikit')
    ax[0].set_title('Sklearn LE: {t:.2g}'.format(t=t1-t0))

    # my Laplacian Eigenmaps algorithm

    t0 = time()
    ml_model = LaplacianEigenmaps(n_components=n_components,
                                  n_neighbors=n_neighbors)
    ml_model.fit(X)
    Y = ml_model.fit_transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='My LE Algorithm')
    ax[1].set_title('My LE: {t:.2g}'.format(t=t1-t0))

    plt.show()

if __name__ == "__main__":
    swiss_roll_test()