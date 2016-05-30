# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:16:28 2016

@author: jemanjohnson
"""

def get_severed_sphere(n_samples, rand_state=0):
    import numpy as np
    from sklearn.utils import check_random_state

    random_state = check_random_state(rand_state)
    # create the sphere
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi
    
    # sever the poles from the sphere
    indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8 ))))
    colors = p[indices]
    X = []
    X[0], X[1], X[3] = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])
        
    return X, colors