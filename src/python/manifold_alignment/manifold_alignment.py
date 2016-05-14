# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:13:26 2016

@author: eman
"""

from __future__ import division

import numpy as np


from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix, block_diag, vstack

from schroedinger_eigenmaps import sim_potential
from graph_construction import create_feature_mat, compute_adjacency, \
                               create_laplacian
from eigenvalue_decomposition import EigSolver



class ManifoldAlignment(object):
    """ This is my manifold alignment object
    """
    def __init__(self,
                 # eigenvalue solver initials
                 eig_solver = 'arpack',
                 norm_lap = False,
                 tol = 1E-12,
                 
                 # eigenvalue tuner initials
                 feature = False,
                 alpha = 17.78,
                 mu = 1.0,
                 beta = 1.0,
                 
                 # knn problem initials
                 n_neighbors = 2,
                 nn_algo = 'brute',
                 metric = 'euclidean',
                 n_jobs = 1,
                 affinity = 'nearest_neighbors',
                 weight = 'heat',
                 gamma = 1.0,
                 trees = 10,
                 
                 # potential matrix initials
                 normalization = 'dis',
                 sp_neighbors = 4,
                 sp_affinity = 'heat',
                 eta = 1.0,
                 
                 # manifold alignment problem parameters
                 ma_method = 'wang11',
                 n_components = 2,
                 
                 # general problem parameters
                 sparse_mat = False,
                 lap_method = 'personal',
                 random_state=0):
         # eigenvalue solver initialization            
         self.n_components = n_components
         self.eig_solver = eig_solver
         self.norm_lap = norm_lap
         self.tol = tol
         
         # eigenvalue tuner initialization
         self.feature = feature
         self.normalization = normalization
         self.alpha = alpha
         self.beta = beta
         self.mu = mu
         
         # knn problem initialization
         self.n_neighbors = n_neighbors
         self.nn_algo = nn_algo
         self.weight = weight
         self.metric = metric
         self.n_jobs = n_jobs
         self.affinity = affinity
         self.gamma = gamma
         self.trees = trees
         
         # potential matrix initials
         self.sp_neighbors = sp_neighbors
         self.sp_affinity = sp_affinity
         self.eta = eta
         
         # manifold alignment problem initialization
         self.ma_method = ma_method

         
         # general problem parameters
         self.sparse_mat = sparse_mat
         self.lap_method = lap_method
         self.random_state = random_state
         
         
    def fit(self, X, Y):
         
        """Fit function for the alignment scheme. This is where I 
        construct all of the necessary matrices
        
        """
        
        # check inputs for specified MA class criteria
        # TODO: check X, Y input as dictionary class
        # TODO: check X, Y keys as list class
        # TODO: check X, Y keys lists are nonempty
        # TODO: check X, Y for specific keys and nothing else
        
        # extract the data from the x, y data inputs
        self.X_data_, X_datasets, \
        self.Y_data_, Y_datasets \
            = self._data(X, Y)
        
        # compute the geometric laplacian term
        L_g, D_g, W_g, W_datasets = self._compute_weights(X_datasets)
        
        # find the approapriate potential matrices
        V_s, V_d = self._compute_potential()

        _, self.embedding_ = self._spectral_embedding()
        
        
        return self
        
    def transform(self, X, n_components=None):
        
        # find the projection functions
        self.eigVecs_ = self._find_projections(X['label'],
                                               n_components=n_components)

        Xproj = {}
        # project the training samples w/ new embedding
        Xproj['train'] = self._project_data(X['label'])
                                   
        # project test samples with new embedding
        Xproj['test'] = self._project_data(X['test'])
        # project data
        
        return Xproj
        
    def _find_projections(self, X, n_components=None):
        # find the number of projected components
        if n_components:
            proj_components = n_components
        else:
            proj_components = self.n_components
        # get the eigenvectors associated with each dataset
        E = []
        start_idx = 0

        for end_idx in self.d_dims_['dataset']:
            
            end_idx += start_idx
            E.append(self.embedding_[start_idx:end_idx,:proj_components+1])
            start_idx += end_idx
        
        # return a list of projection functions
        return E
        
    def _project_data(self, X):
        
         return [np.dot(XeigVec.T, Xdata.T).T for Xdata, XeigVec \
                     in zip(X,self.eigVecs_)]

    
    def _spectral_embedding(self):
        # TODO: create feature-based implementation
        # tune the eigenvalue problem
        A, B = self._embedding_tuner()

        # Fit the parameters to solve the generalized eigenvalue problem
        eig_model = EigSolver(n_components=self.n_components,
                              eig_solver=self.eig_solver,
                              sparse=self.sparse_mat,
                              tol=self.tol,
                              norm_laplace=self.norm_lap)
        if not self.sparse_mat:
            return eig_model.find_eig(A=A.toarray(), B=B.toarray())
        else:
            return eig_model.find_eig(A=A, B=B)
            

        
    def _embedding_tuner(self):
        

        if self.ma_method == 'wang11':
            
            A = self.L_g + self.mu * self.V_s
            B = self.V_d
            
        elif self.ma_method in ['wang']:
            
            A = self.L_g + self.mu * self.V_s
            B = self.D_g
        
        elif self.ma_method in ['ssma']:
            
            A = self.mu * self.L_g + (1- self.mu) * self.V_s
            B = self.V_d
        
        elif self.ma_method in ['sema']:
            
            A = self.L_g + self.mu * self.V_s
            B = self.D_g + self.V_d
        
        else:
            raise ValueError('Unrecognized Manifold Alignment problem.')

        return create_feature_mat(self.X_data_, A, sparse=True), \
                create_feature_mat(self.X_data_, B, sparse=True)
    
    def _data(self, X, Y):
        
        # get dimensions of the input variables
        self.n_samples_, self.d_dims_ = self._get_dimensions(X, Y)
        
        # create X['dataset'] key with stacks of labeled and 
        # unlabeled data
        
        # vertically concatenate labeled and unlabeled data
        X_datasets = create_stacks(X['label'], X['unlabel'])
        
        self.X_train = X['label']
        # find the similarity and dissimilarity matrices
        #TODO: figure out how to stack a list of sparse objects
        # on a list of arrays. (eg label (2xmxn), unlabel (2))

        Y_unlabel = create_unlabeled(X['unlabel'], sparse=None)
        
        Y_datasets = create_stacks(Y['label'], Y_unlabel)
        
        # create block diagonal matrices from datasets
        # TODO: create feature-based option
        return block_diag(X_datasets), X_datasets, \
                coo_matrix(np.vstack(Y_datasets)),  Y_datasets
        
    def _compute_potential(self, 
                           X=None, 
                           X_datasets=None,
                           X_datasets_imgs=None):
        
        # construct the similarity and dissimilarity potential matrix
        if self.ma_method in ['wang', 'ssma', 'wang11', 'kema', 'sema']:
            
            # compute the similarity potential
            self.V_s, _ = sim_potential(self.Y_data_,
                                     potential='sim',
                                     norm_lap=self.norm_lap,
                                     method=self.lap_method,
                                     sparse_mat=self.sparse_mat)
                                     
            # compute the dissimilarity potential
            self.V_d, _ = sim_potential(self.Y_data_,
                                        potential='dis',
                                        norm_lap=self.norm_lap,
                                        method=self.lap_method,
                                        sparse_mat=self.sparse_mat)
            
                                     
            
            # TODO: create spatial-spectral potential
            '''Any method w/ Spatial-Spectral Potential
            will have to loop through each of the laplacian
            datasets and construct spatial-spectral terms for each 
            individually. Then block them all together.
            Parameters - X, X_datasets, X_images
             '''
        else:
            raise ValueError('Unrecognized manifold alignment method.')
         
        return self.V_s, self.V_d
            
        
        
    def _adjacency_blocks(self, X):
        """Create adjacency block matrices from adjacency matrices
    
        Parameters
        ----------
        X               - a k list of (n x m) dense array entries
        
        Returns
        -------
        W               - a sparse diagonal (n*k) x (n*k) matrix 
        W_datasets      - a list of sparse (n x m) adjacency matrix entries           
        """

        
        # create adjacency matrices        
        W_datasets = [compute_adjacency(dataset,
                                        n_neighbors=self.n_neighbors,
                                        affinity=self.affinity,
                                        weight=self.weight,
                                        sparse=self.sparse_mat,
                                        neighbors_algorithm=self.nn_algo,
                                        metric=self.metric,
                                        trees=self.trees,
                                        gamma=self.gamma) \
                                            for dataset in X]
        
        ''' TODO: list comprehensions of list comprehensions for different
        values in the adjacency matrix. e.g. different weight, affinity,
        n_neighbors, metric, trees, gamma, algorithm for each matrix.
        '''
        # return 
        return block_diag(W_datasets), W_datasets

        
    
    def _laplacian_blocks(self, Adjacency):
        """Creates block diagonal matrices from adjacency matrices
        
        Parameters:
        -----------
        W_datasets          - a list of k (n x n) sparse matrices
        
        Returns
        -------
        L                   - a sparse (n*k x n*k) Laplacian matrix 
        L_datasets          - a list of k (n x n) sparse Laplacian matrices
        D                   - a sparse (n*k x n*k) Diagonal matrix
        D_datasets          - a list of k (n x n) sparse Diagonal matrices
        
        References
        ----------
        1) Clever trick with the dual output list comprehension
            stackoverflow: 
                http://goo.gl/ojpRQg
                
            It don't think it's faster to compute the lists separately 
            because I'm doing some heavier calculations behind the scenes
            as compared to actual number creating.
        """

        
        L_datasets, D_datasets = zip(*[create_laplacian(dataset,
                                      norm_lap=self.norm_lap,
                                      method=self.lap_method,
                                      sparse=self.sparse_mat) \
                                                   for dataset in Adjacency])
        
        return block_diag(L_datasets), L_datasets, \
                block_diag(D_datasets), D_datasets  
    def _compute_weights(self, X_datasets):
        """ Creates the Weighted adjacency matrix
        """
        
        # create adjacency block matrices
                # compute the geometric adjacency matrix from the data
        self.W_g, W_datasets = self._adjacency_blocks(X_datasets)
        
        # create Laplacian and Degree block matrices
        self.L_g, L_datasets, self.D_g, D_datasets \
            = self._laplacian_blocks(W_datasets)
            
        return self.L_g, self.D_g, self.W_g, W_datasets 
                

        
        
        
    def _get_dimensions(self, X, Y):
        """This function will extract the number of samples and number of
        dimensions from the X and Y values.
        
        Parameters:
        ----------
        X    - dictionary of values
            e.g. X['labeled']       -> a list of values
                 X['unlabeled']     -> a list of values
                 
        Returns
        -------
        n_samples : dictionary, keys - 'label', 'unlabel', 'data', 'dataset'
        """
        # initialize data structures for saving the variables
        n_samples = {}; d_dims = {}
        n_samples['label'] = []; d_dims['label'] = []
        n_samples['unlabel'] = []; d_dims['unlabel'] = []
        n_samples['dataset'] = []; d_dims['dataset'] = []
        n_samples['test'] = []; d_dims['test'] = []
        n_samples['data'] = []; d_dims['data'] = []
        
        
        
        for labeled_data, unlabeled_data, test_data in zip(X['label'],
                                                           X['unlabel'],
                                                           X['test']):
        
            # grab dimensions from labeled data strucutres
            n_labeled, d_labeled = np.shape(labeled_data)
            
            # save values
            n_samples['label'].append(n_labeled)
            d_dims['label'].append(d_labeled)
            
            # grab dimensions from unlabeled data structures
            n_unlabeled, d_unlabeled = np.shape(unlabeled_data)
            
            # save values
            n_samples['unlabel'].append(n_unlabeled)
            d_dims['unlabel'].append(d_unlabeled)
            
            # save labeled and unlabeled values
            n_samples['dataset'].append(n_labeled+n_unlabeled)
            d_dims['dataset'].append(d_labeled)
        
            # grab dimensions from unlabeled data structures
            n_test, d_test = np.shape(test_data)
            n_samples['test'].append(n_test)
            d_dims['test'].append(d_test)
            
        n_samples['data'] = np.sum(n_samples['dataset'])
        d_dims['data'] = np.sum(d_dims['dataset'])
        
        return n_samples, d_dims
        
# create matrix stacks
def create_stacks(X1, X2, sparse=None):
    """Utility function - horizontally stacks two lists together. Assumes 
    that the two list entries are (n x m) and (n' x m') where m=m' but not
    necessarily n = n'.
    
    Parameters:
    ----------
    X1              - a list of (n x m) entries
    X2              - a list of (n' x m) entries
    sparse_mat      - determines whether the arrays are dense or sparse
                      (coo). (default=None)
    
    Returns:
    --------
    X               - a list of (n + n') x (m) entries
    """
    
    # create an empty list    
    if sparse:                  # sparse arrays creation
        X = [vstack([x1, x2], format='coo') for \
             x1, x2 in zip(X1, X2)]
    
    else:                           # dense arrays creation
        X = [np.vstack([x1, x2]) for x1, x2 in zip(X1, X2)]
    
    return X   

# create labeled data matrix
def create_unlabeled(X, sparse=None):
    """Utility function that creates a vector of all zeros that mimics 
    the unlabeled samples as a list.
    
    Parameters:
    ----------
    X         - a list of (nxm) entries of data points
    
    Returns:
    --------
    Y         - a list of (nxm) entries of zeros
    """
 
    if sparse:              # sparse array creation
        return  [coo_matrix((np.shape(data)[0], 1),
                                  dtype=np.float) for data in X]
    else:
        return [np.zeros((np.shape(data)[0], 1),
                         dtype=np.float) for data in X]


# my test function to see if it works. Should be in the 80s range of
# accuracy
def test_ma_gaussian(ma_method='wang', n_components=2):
    
    from data_generation import generate_gaussian
    
    # define some dictionaries with empty labeled lists
    X ={}; Y={};
    X['label'] = []; X['unlabel'] = []; X['test'] = []
    Y['label'] = []; Y['unlabel'] = []; Y['test'] = []
    
    
    # assign labels from gaussian dataset
    X1, X2, XT1, XT2, \
    Y1, Y2, YT1, YT2, \
    U1, U2 = generate_gaussian()
    
    
    # create appropriate data structures based off of
    # the manifold alignment class criteria
    X['label'] = [X1, X2]
    X['unlabel'] = [U1, U2]
    X['test'] = [XT1, XT2]
    Y['label'] = [Y1 , Y2]
    Y['test'] = [YT1, YT2] 
    
    print np.shape(X['label'][0]), np.shape(Y['label'][0])
    print np.shape(X['unlabel'][0])
    print np.shape(X['test'][0]), np.shape(Y['test'][0])
    
    print np.shape(X['label'][1]), np.shape(Y['label'][1])
    print np.shape(X['unlabel'][1])
    print np.shape(X['test'][1]), np.shape(Y['test'][1])    
    
    ma_method = ManifoldAlignment(ma_method=ma_method,
                                  lap_method='personal')
    ma_method.fit(X,Y)
    Xproj = ma_method.transform(X, n_components=2)
    
    from classification_list import lda_pred, accuracy_stats
    Y['pred'] = lda_pred(Xproj['train'],
                     Xproj['test'],
                     Y['label'],
                     Y['test'])

    Acc_stats = accuracy_stats(Y['pred'], Y['test'])
    

    Lg = ma_method.L_g
    Vs = ma_method.V_s
    Vd = ma_method.V_d
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(nrows=3, ncols=1,
                           figsize=(10,10))

    ax[0].spy(Lg, precision=1E-5, markersize=.2)
    ax[0].set_title('Geometric Laplacian')
    ax[1].spy(Vs, precision=1E-5, markersize=.2)
    ax[1].set_title('Similarity Potential')
    ax[2].spy(Vd, precision=1E-5, markersize=.2)
    ax[2].set_title('Dissimilarity Potential')
    
    plt.show()
    
    print('AA - Domain 1: {s}'.format(s=Acc_stats['AA'][0]))
    print('AA - Domain 2: {s}'.format(s=Acc_stats['AA'][1]))
    

if __name__ == "__main__":
    
    test_ma_gaussian(ma_method='ssma', n_components=3)

    

