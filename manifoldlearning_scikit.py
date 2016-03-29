# -*- coding: utf-8 -*-
# Author: Juan E Johnson --<emanjohnson91@gmail.com>
"""
This is just implementing the standard manifold alignment algorithms that
are found in the scikit learn library. I want to get a feel for them and
try them on some artificial datasets:
* s curve
* swiss roll
* severed sphere

Original Code by: Jake Vanderplas
Link: http://goo.gl/ePMFeC
"""
print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# generate the dataset 

n_points = 1000                    # number of points
X, color = datasets.samples_generator.make_s_curve(n_points, 
                                                   random_state=0)
                                                 
n_neighbors = 10        # number of neighbours
n_components = 2        # number of components
#----------
# plot data
#----------
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=color, cmap=plt.cm.spectral)

#---------------------
# Laplacian Eigenmaps
#---------------------
t0 = time()
LE = manifold.SpectralEmbedding(n_components=n_components,
                                affinity='rbf',
                                n_neighbors=n_neighbors)                
Y = LE.fit_transform(X)
t1 = time()

print("Laplacian Eigenmaps: {n:.2g} sec".format(n=t1-t0))

#---------------
# plot embedding
#---------------
ax = fig.add_subplot(122)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')




