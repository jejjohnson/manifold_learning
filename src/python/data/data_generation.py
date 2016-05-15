# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:44:10 2016

@author: eman
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian(N=10,
                      U=200,
                      T=500,
                      plot_data=False,
                      mirror=False,
                      square=False):
    #---------
    # Domain I
    #---------
    mean1 = np.array([-1, -1], dtype='float')
    mean2 = np.array([-1,-2], dtype='float')
    cov = np.array(([1, .9], [.9, 1]), dtype='float')

    # Generate a Gaussian dataset from the parameters - mean, covariance, variance
    X1class1 = np.random.multivariate_normal(mean1, cov, N)
    X1class2 = np.random.multivariate_normal(mean2, cov, N)
    X1 = np.vstack((X1class1, X1class2))
    Y1 = np.ones((2*N,1), dtype='float')
    Y1[N:,:] = 2

    # Unlabeled Dataset
    U1class1 = np.random.multivariate_normal(mean1,cov,np.int(U/2))
    U1class2 = np.random.multivariate_normal(mean2,cov,np.int(U/2))
    U1 = np.vstack((U1class1, U1class2))
    Y1U = np.zeros((U,1), dtype='float')

    # Testing Dataset
    XT1class1 = np.random.multivariate_normal(mean1, cov, np.int(T/2))
    XT1class2 = np.random.multivariate_normal(mean2, cov, np.int(T/2))
    XT1 = np.vstack((XT1class1, XT1class2))
    YT1 = np.ones((T,1), dtype='float')
    YT1[np.int((T/2)):,:] = 2

    # Invert the X-Axis of Domain 1

    if mirror:
        X1[0,:] = X1[0,:] * -1
        U1[0,:] = U1[0,:] * -1
        XT1[0,:] = XT1[0,:] * -1

    if square:
        X1[0,:] = X1[0,:] ** 2
        U1[0,:] = U1[0,:] ** 2
        XT1[0,:] = XT1[0,:] ** 2



    #----------
    # Domain II
    #----------
    mean1 = np.array([3, -1], dtype='float')
    mean2 = np.array([3,-2], dtype='float')
    cov = np.array(([1, .9], [.9, 1]), dtype='float')

    # Generate a Gaussian dataset from the parameters - mean, covariance, variance
    X2class1 = np.random.multivariate_normal(mean1, cov, N)
    X2class2 = np.random.multivariate_normal(mean2, cov, N)
    X2 = np.vstack((X2class1, X2class2))

    Y2 = np.ones((2*N,1), dtype='float')
    Y2[N:,:] = 2

    U2class1 = np.random.multivariate_normal(mean1,cov,np.int(U/2))
    U2class2 = np.random.multivariate_normal(mean2,cov,np.int(U/2))
    U2 = np.vstack((U2class1, U2class2))
    Y2U = np.zeros((U,1), dtype='float')

    XT2class1 = np.random.multivariate_normal(mean1, cov, np.int(T/2))
    XT2class2 = np.random.multivariate_normal(mean2, cov, np.int(T/2))
    XT2 = np.vstack((XT2class1, XT2class2))

    YT2 = np.ones((T,1), dtype='float')
    YT2[np.int((T/2)):,:] = 2

    if plot_data:


        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.plot(U1[:,0], U1[:,1], 'g.')
        ax1.scatter(XT1[:,0], XT1[:,1], s=10,c=YT1)
        ax1.scatter(X1[:,0], X1[:,1], s=30, c=Y1)
        ax1.set_title('Dataset I')

        ax2.plot(U2[:,0], U2[:,1], 'g.')
        ax2.scatter(XT2[:,0], XT2[:,1], s=10, c=YT2)
        ax2.scatter(X2[:,0], X2[:,1], s=30, c=Y2)
        ax2.set_title('Dataset I')
        plt.show()


    return X1, X2, XT1, XT2, Y1, Y2, YT1, YT2, U1, U2
