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

# import necessary packages
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

# generate the dataset 

n_points = 1000                    # number of points
X = {}; colors = {}
data, color = datasets.samples_generator.make_s_curve(n_points,random_state=0)
X['scurve'] = data; colors['scurve'] = color
del data; del color
data, color = datasets.samples_generator.make_swiss_roll(n_points,
                                                         noise=.5,
                                                         random_state=0)
X['swissroll'] = data; colors['swissroll'] = color

del data; del color
import get_data as gt
data, color = gt.get_severed_sphere(n_points,
                                    rand_state=0)
X['sphere'] = data; colors['sphere'] = color
                                      

#%%----------
# plot data
#----------
                                                 
dataset = 'sphere'

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[dataset][:,0], X[dataset][:,1], X[dataset][:,2],
           c=color,
           cmap=plt.cm.spectral)
plt.title("Original Dataset: %s" % dataset)

#%%---------------------
# Laplacian Eigenmaps
#---------------------

# important parameters
n_neighbors = 10        # number of neighbours
n_components = 2        # number of components
kernel = 'nearest_neighbors'          # kernel for affinity matrix

t0 = time()             # start timer
LE = manifold.SpectralEmbedding(n_components=n_components,
                                affinity=kernel,
                                n_neighbors=n_neighbors)                
X['transform'] = LE.fit_transform(X[dataset])
t1 = time()             # stop timer

print("Laplacian Eigenmaps: {n:.2g} sec".format(n=t1-t0))

#%%---------------
# plot embedding
#---------------
ax = fig.add_subplot(122)
plt.scatter(X['transform'][:, 0],
            X['transform'][:, 1],
            c=color, cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')




