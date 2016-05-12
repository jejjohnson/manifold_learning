"""
Eigenvalue Decompositions using different methods found 
from Python packages and codes
"""
# Authors: Eman Johnson
# License: BSD 3 clause

from __future__ import division
import warnings
import numpy as np
#import numpy.linalg as linalg
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import lobpcg, eigs, eigsh
from scipy.linalg import eigh
from scipy import linalg
from sklearn.utils import check_array


class EigSolver(object):
    """A class of eigenvalue decomposition algorithms.
    
    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold
    
    eig_method : str ['dense'|'robust'|'arpack'|'multi']
        some methods to choose from when solving the eigenvalue 
        decomposition problem
    
    TODO: 'rsvd'
    TODO: better functions to capture variables
    """
    def __init__(self,
                 n_components = 2,
                 eig_solver = 'dense',
                 sparse = False,
                 tol = 1.E-12,
                 norm_laplace=False):
         self.n_components = n_components
         self.eig_solver = eig_solver
         self.sparse = sparse
         self.tol = tol
         self.norm_laplace = norm_laplace

         
    def find_eig(self, A, B=None):
         
         if self.sparse and self.eig_solver not in ['arpack', 'multi']:
             self.eig_solver = 'arpack'
         elif not self.sparse and self.eig_solver not in ['robust', 'dense']:
             self.eig_solver = 'dense'

         if self.eig_solver == 'robust' and not self.sparse:
             
             eigVals, eigVecs = eigh_robust(a=A, b=B,
                                        eigvals=(0, self.n_components-1))


                 
         elif self.eig_solver == 'dense':
             
             eigVals, eigVecs = eig_dense(A=A, B=B, 
                                          k_dims=self.n_components)
         elif self.eig_solver == 'arpack':
            
            eigVals, eigVecs =  eig_scipy(A=A,
                                          B=B,
                                          n_components=self.n_components)
                                               
         elif self.eig_solver == 'multi':
            
            eigVals, eigVecs = eig_multi(A=A,
                                         B=B,
                                          n_components=self.n_components, 
                                          tol=self.tol)
                                          
         elif self.eig_solver == 'rsvd':
             _, eigVals, eigVecs = r_svd(M=A,
                                            n_components=self.n_components)
         else:
             raise ValueError('Unrecognizable Eigenvalue Method.')
             
         return eigVals, eigVecs
         
         
         

#--------------------------------------
# Scipy - ARPACK Dense (small)
#--------------------------------------
def eig_dense(A, B=None, k_dims=2):
    
    return eigh(a=A,
                b=B,
                eigvals=(1, k_dims),
                type=1)
    
#--------------------------------------
# Scipy - ARPACK Sparse
#--------------------------------------
def eig_scipy(A, B=None, n_components=2+1, method='arpack'):
    """

    """
    # There is a bug for a low number of nodes for this solver
    # calculate more eigenvalues than necessary
    if n_components <= 10:
        numEigs = n_components+15
    else:
        numEigs = n_components

    # check to make sure the number of eigs is less than dim of A
    assert numEigs <= A.shape[0], \
    'Number of components less than or equal to A'

    # Solve using the eigenvale method
    eigenvalues, eigenvectors = eigsh(A=A,
                                          k=numEigs,
                                          M=B,
                                          which='SM')

    return eigenvalues[:n_components+1], eigenvectors[:,:n_components+1]

#--------------------------------------
# Pyamg - Multigrid
#--------------------------------------
def eig_multi(A, B=None, n_components=2+1, tol=1E-12):
    """Solves the generalized Eigenvalue problem:
    A x = lambda B x using the multigrid method.
    Works well with very large matrices but there are some
    instabilities sometimes.
    """
    # convert matrix A and B to float
    A = A.astype(np.float64); 
    
    if B is not None:
        B = B.astype(np.float64)
    
    # import the solver
    ml = smoothed_aggregation_solver(A)
    
    # preconditioner
    M = ml.aspreconditioner()
    
    # initial guess for X
    np.random.RandomState(seed=1234)
    X = np.random.rand(A.shape[0], n_components+1)
    
    # solve using the lobpcg algorithm
    eigenvalues, eigenvectors = lobpcg(A, X, M=M, B=B,
                                       tol=tol,
                                       largest='False')
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:,::-1]
    return eigenvalues[:n_components+1], eigenvectors[:,:n_components+1]



#-------------------------
# Github - Randomized SVD
#-------------------------
def r_svd(M, n_components=2+1):
   """
   Computes Gunnar Martinsson's Fast SVD
   Source Code: https://gist.github.com/alextp/662433

   Parameters
   ----------
   M: ndarray or sparse matrix
        Matrix to decompose

   n_components: int
        Number of singular values and vectors to extract
   """
   p = n_components+5
   Y = np.dot(M, np.random.normal(size=(M.shape[1],p)))
   Q, r = np.linalg.qr(Y)
   B = np.dot(Q.T, M)
   Uhat, s, v = np.linalg.svd(B, full_matrices=False)
   U = np.dot(Q, Uhat)
   return U.T[:n_components+1].T, s[:n_components+1], v[:n_components+1]

#---------------------------
# robust eigenvalue problem
#---------------------------

def eigh_robust(a, b=None, eigvals=None, eigvals_only=False,
                overwrite_a=False, overwrite_b=False,
                turbo=True, check_finite=True):
    """Robustly solve the Hermitian generalized eigenvalue problem
    This function robustly solves the Hermetian generalized eigenvalue problem
    ``A v = lambda B v`` in the case that B is not strictly positive definite.
    When B is strictly positive-definite, the result is equivalent to
    scipy.linalg.eigh() within floating-point accuracy.
    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric matrix.
        If omitted, identity matrix is assumed.
    eigvals : tuple (lo, hi), optional
        Indexes of the smallest and largest (in ascending order) eigenvalues
        and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
        If omitted, all eigenvalues and eigenvectors are returned.
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    turbo : bool, optional
        Use divide and conquer algorithm (faster but expensive in memory,
        only for generalized eigenvalue problem and if eigvals=None)
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in `b` (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    w : (N,) float ndarray
        The N (1<=N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.
    v : (M, N) complex ndarray
        (if eigvals_only == False)
    """
    kwargs = dict(eigvals=eigvals, eigvals_only=eigvals_only,
                  turbo=turbo, check_finite=check_finite,
                  overwrite_a=overwrite_a, overwrite_b=overwrite_b)

    # Check for easy case first:
    if b is None:
        return linalg.eigh(a, **kwargs)

    # Compute eigendecomposition of b
    kwargs_b = dict(turbo=turbo, check_finite=check_finite,
                    overwrite_a=overwrite_b)  # b is a for this operation
    S, U = linalg.eigh(b, **kwargs_b)

    # Combine a and b on left hand side via decomposition of b
    S[S <= 0] = np.inf
    Sinv = 1. / np.sqrt(S)
    W = Sinv[:, None] * np.dot(U.T, np.dot(a, U)) * Sinv
    output = linalg.eigh(W, **kwargs)

    if eigvals_only:
        return output
    else:
        evals, evecs = output
        return evals, np.dot(U, Sinv[:, None] * evecs)
        

