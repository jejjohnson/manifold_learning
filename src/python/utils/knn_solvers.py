# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:39:17 2016

@author: eman
"""
from sklearn.neighbors import NearestNeighbors, LSHForest
from annoy import AnnoyIndex
# import hdidx
import numpy as np

class KnnSolver(object):

    def __init__(self,
                 # knn parameters
                 n_neighbors = 2,
                 nn_algorithm = 'brute',
                 metric = 'euclidean',
                 n_jobs = 1,
                 affinity = 'nearest_neighbor',
                 weight = 'heat',
                 gamma = 1.0,
                 p_norm = 2,

                 # ball tree and kdtree parameters
                 trees = 10,
                 leaf_size = 30,

                 # scikit LSHForest parameters
                 n_estimators = 10,
                 min_hash_match = 4,
                 n_candidates = 10,
                 random_state = 0):
      self.n_neighbors = n_neighbors
      self.nn_algorithm = nn_algorithm
      self.metric = metric
      self.n_jobs = n_jobs
      self.affinity = affinity
      self.weight = weight
      self.gamma = gamma
      self.trees = trees
      self.leaf_size = leaf_size
      self.p_norm = p_norm

      # scikit LSHForest parameters
      self.n_estimators = trees
      self.min_hash_match = min_hash_match
      self.n_candidates = n_candidates
      self.random_state = random_state

    def find_knn(self, data):

       if self.nn_algorithm in ['brute', 'kd_tree', 'ball_tree']:

           return knn_scikit(data,
                             n_neighbors=self.n_neighbors,
                             leaf_size = self.leaf_size,
                             metric = self.metric,
                             p = self.p_norm)
       elif self.nn_algorithm in ['lshf']:

           return lshf_scikit(data,
                              n_neighbors=self.n_neighbors,
                              n_estimators=self.n_estimators,
                              min_hash_match=self.min_hash_match,
                              n_candidates=self.n_candidates,
                              random_state=self.random_state)
       elif self.nn_algorithm in ['annoy']:

           return ann_annoy(data,
                            metric=self.metric,
                            n_neighbors=self.n_neighbors,
                            trees=self.trees)

       elif self.nn_algorithm in ['hdidx']:
           raise NotImplementedError('Unrecognized K-Nearest Neighbor Method.')
       #
       #     return ann_hdidx(data,
       #                      n_neighbors = self.n_neighbors,
       #                      indexer=self.trees)
       else:
           raise ValueError('Unrecognized NN Method.')

# sklearns nearest neighbors formula
def knn_scikit(data, n_neighbors=4,
               algorithm='brute',
               leaf_size=30,
               metric='euclidean',
               p=None):
   n_neighbors += 1

   # initialize nearest neighbor model
   nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                           algorithm=algorithm,
                           leaf_size=leaf_size,
                           metric=metric,
                           p=p)

   # fit nearest neighbor model to the data
   nbrs.fit(data)

   # return the distances and indices
   return nbrs.kneighbors(data)

# scikit learns locality sensitive hashing function
def lshf_scikit(data, n_neighbors=4,
               n_estimators=10,
               min_hash_match=4,
               n_candidates=10,
               random_state=None):
   n_neighbors += 1

   # initialize nearest neighbor model
   nbrs = LSHForest(n_neighbors=n_neighbors,
                    n_estimators = 10,
                    min_hash_match = 4,
                    n_candidates = 10,
                    random_state = 0)

   # fit nearest neighbor model to the data
   nbrs.fit(data)

   # return the distances and indices
   return nbrs.kneighbors(data)

# annoy approximate nearest neighbor function
def ann_annoy(data, metric='euclidean',
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

# Hdidx package for approximate nearest neighbor function
# def ann_hdidx(data,
#               n_neighbors = 10,
#               indexer=8):
#
#    datapoints = data.shape[0]
#    dimensions = data.shape[1]
#
#    data_query = np.random.random((n_neighbors, dimensions))
#    print np.shape(data_query)
#
#    # create Product Quantization Indexer
#    idx = hdidx.indexer.IVFPQIndexer()
#
#    # build indexer
#    idx.build({'vals': data, 'nsubq': indexer})
#
#    # add database items to the indexer
#    idx.add(data)
#
#    # searching in the database and return top-10 items for
#    # each query
#    idn, distVals = idx.search(data, n_neighbors)
#    return distVals, idn

if __name__ == "__main__":

    import numpy as np
    import time as time

    n_dims = 200
    n_samples = 5000

    X_data = np.random.random((n_samples, n_dims))

    print('Size of X is {s}'.format(s=np.shape(X_data)))


    for nn_model in ['brute','kd_tree', 'ball_tree', 'annoy']:

        t0 = time.time()
        # initialize knn model
        knn_model = KnnSolver(nn_algorithm=nn_model, n_jobs=5,
                              n_neighbors=20)

        # fit knn model to the data
        distVals, idx = knn_model.find_knn(X_data)
        t1 = time.time()

        print('{m}, time taken: {s:.2f}'.format(m=nn_model, s=t1-t0))
