# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from img_preprocessing_techniques import img_as_array, img_gt_idx
from potential_functions import get_spatial_coordinates
from data_filegrab import get_hsi  

# grabbing data function    
def data_grab(dataset='gaussian',
              train_size=.2,
              label_size=.2,
              plot_data=None):
    
    # define some empty sets
    X ={}; Y={};
    X['label'] = []; X['unlabel'] = []; X['test'] = []
    Y['label'] = []; Y['unlabel'] = []; Y['test'] = []
    
    
    if dataset in ['gaussian']:
        # assign labels from gaussian dataset
        X1, U1, Y1, Y1U, XT1, YT1, X2, U2, Y2, Y2U, XT2, YT2 \
        = gaussian_data()
        
    if dataset in ['hsi']:
        
        X['spatial'] = []
        X1, U1, Y1, XT1, YT1, X2, U2, Y2, XT2, YT2, X1sp, X2sp\
        = grab_hsi(train_size=train_size, 
                   label_size=label_size,
                   plot_data=plot_data)
        
        # create appropriate data structures
        X['label'].append(X1); X['label'].append(X2)
        X['unlabel'].append(U1); X['unlabel'].append(U2)
        X['test'].append(XT1); X['test'].append(XT2)
        Y['label'].append(Y1); Y['label'].append(Y2)
        Y['test'].append(YT1); Y['test'].append(YT2)
        X['spatial'].append(X1sp); X['spatial'].append(X2sp)
        
    else:
        raise ValueError('Sorry. Need a valid dataset generation')
    

    
    return X, Y

      
#--------------
# HSI Data 
#--------------
def grab_hsi(train_size=.2, label_size=.2, plot=None):
    
    img_data = get_hsi() 
    
    img1 = img_data['img1']
    img2 = img_data['img2']
    gt1 = img_data['gt1']
    gt2 = img_data['gt2']

    
    # reshape arrays
    imgVec1 = img_as_array(img1); imgVec2 = img_as_array(img2)
    gtVec1 = img_as_array(gt1, gt=True); gtVec2 = img_as_array(gt2, gt=True)
    
    # get spatial vector
    pData1 = get_spatial_coordinates(img1)
    pData2 = get_spatial_coordinates(img2)
    
    # pair the X and y indices
    X1, Y1 = img_gt_idx(imgVec1, gtVec1)
    X2, Y2 = img_gt_idx(imgVec2, gtVec2)
    
    #----------------------------
    # Training and Testing
    #----------------------------
    
    test_size = 1-train_size
    # image 1
    sss = StratifiedShuffleSplit(Y1, 1, test_size=test_size)
    for train_idx, test_idx in sss:
        X1_train, Y1_train = X1[train_idx], Y1[train_idx]
        X1_test, Y1_test = X1[test_idx], Y1[test_idx]
        pData1 = pData1[train_idx]
        
    # image 2
    sss = StratifiedShuffleSplit(Y2, 1, test_size=test_size)
    for train_idx, test_idx in sss:
        X2_train, Y2_train = X2[train_idx], Y2[train_idx]
        X2_test, Y2_test = X2[test_idx], Y2[test_idx]
        pData2 = pData2[train_idx]
        
    
    #-----------------------
    # Labeled and Unlabeled
    #-----------------------
    
    unlabel_size = 1-label_size
    
    # image 1
    sss = StratifiedShuffleSplit(Y1_train, 1, test_size=unlabel_size)
    
    for label_idx, unlabel_idx in sss:
        X1_label, Y1_label = X1_train[label_idx], Y1_train[label_idx]
        X1_unlabel, Y1_unlabel = X1_train[unlabel_idx], Y1_train[unlabel_idx]
        
    
    pData1 = np.vstack((pData1[label_idx], pData1[unlabel_idx]))
    # image 2 
    sss = StratifiedShuffleSplit(Y2_train, 1, test_size=unlabel_size)
       
    for label_idx, unlabel_idx in sss:
        X2_label, Y2_label = X2_train[label_idx], Y2_train[label_idx]
        X2_unlabel, Y2_unlabel = X2_train[unlabel_idx], Y2_train[unlabel_idx] 
    
    pData2 = np.vstack((pData2[label_idx], pData2[unlabel_idx]))    
    
    return X1_label, X1_unlabel, Y1_label, X1_test, Y1_test, \
           X2_label, X2_unlabel, Y2_label, X2_test, Y2_test, \
           pData1, pData2
    



# generate gaussian data
def gaussian_data(plot=None):
    
    # Generate Sample Data
    N = 10                    # Labeled Samples per and Domain
    U = 200                   # Unlabeled Samples per Domain
    T = 500                   # Test Samples per Domain
    
    #---------
    # domain I
    #---------
    
    mean1 = np.array([-1, -1], dtype='float')
    mean2 = np.array([-1,-2], dtype='float')
    cov = np.array(([1, .9], [.9, 1]), dtype='float')
    
    # Generate a Gaussian dataset from the parameters - mean, cov, var
    X1class1 = np.random.multivariate_normal(mean1, cov, N)
    X1class2 = np.random.multivariate_normal(mean2, cov, N)
    X1 = np.vstack((X1class1, X1class2))
    Y1 = np.ones((2*N,1), dtype='float')
    Y1[N:,:] = 2
    
    # unlabeled data
    U1class1 = np.random.multivariate_normal(mean1,cov,U/2)
    U1class2 = np.random.multivariate_normal(mean2,cov,U/2)
    U1 = np.vstack((U1class1, U1class2))
    Y1U = np.zeros((U,1), dtype='float')
    
    # testing data
    XT1class1 = np.random.multivariate_normal(mean1, cov, T/2)
    XT1class2 = np.random.multivariate_normal(mean2, cov, T/2)
    XT1 = np.vstack((XT1class1, XT1class2))
    YT1 = np.ones((T,1), dtype='float')
    YT1[(T/2):,:] = 2
    
    #---------
    # domain 2
    #---------
    
    mean1 = np.array([3, -1], dtype='float')
    mean2 = np.array([3,-2], dtype='float')
    cov = np.array(([1, .9], [.9, 1]), dtype='float')
    
    # Generate a Gaussian dataset from the parameters - mean, cov, var
    X2class1 = np.random.multivariate_normal(mean1, cov, N)
    X2class2 = np.random.multivariate_normal(mean2, cov, N)
    X2 = np.vstack((X2class1, X2class2))
    
    Y2 = np.ones((2*N,1), dtype='float')
    Y2[N:,:] = 2
    
    U2class1 = np.random.multivariate_normal(mean1,cov,U/2)
    U2class2 = np.random.multivariate_normal(mean2,cov,U/2)
    U2 = np.vstack((U2class1, U2class2))
    Y2U = np.zeros((U,1), dtype='float')
    
    XT2class1 = np.random.multivariate_normal(mean1, cov, T/2)
    XT2class2 = np.random.multivariate_normal(mean2, cov, T/2)
    XT2 = np.vstack((XT2class1, XT2class2))
    
    YT2 = np.ones((T,1), dtype='float')
    YT2[(T/2):,:] = 2
    
    #------------------
    # data deformations
    #------------------
    
    deformations = None
    # Options: 'mirror', 'square'
    
    if deformations == 'square':
        X1[0,:] = X1[0,:]**-1
        U1[0,:] = U1[0,:]**-1
        XT1[0,:] = XT1[0,:]**-1
    elif deformations == 'mirror':
        X1[0,:] = X1[0,:]**2
        U1[0,:] = U1[0,:]**2
        XT1[0,:] = XT1[0,:]**2
        
    #------------------
    # plot data
    #-----------------
    if plot:
        plt.style.use('ggplot')
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(10,5))
        
        ax1.plot(U1[:,0], U1[:,1], 'g.')
        ax1.scatter(XT1[:,0], XT1[:,1], s=10,c=YT1)
        ax1.scatter(X1[:,0], X1[:,1], s=30, c=Y1)
        ax1.set_title('Dataset I')
        
        ax2.plot(U2[:,0], U2[:,1], 'g.')
        ax2.scatter(XT2[:,0], XT2[:,1], s=10, c=YT2)
        ax2.scatter(X2[:,0], X2[:,1], s=30, c=Y2)
        ax2.set_title('Dataset II')
        
        plt.show()    
    
    #-------------
    # return data
    #------------
    
    return X1, U1, Y1, Y1U, XT1, YT1, X2, U2, Y2, Y2U, XT2, YT2
   



    
    
    
    



    
            
        