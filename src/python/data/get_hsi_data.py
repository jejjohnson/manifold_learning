"""
Get the HSI data from the raw folder
"""

import os, inspect
import scipy.io
import spectral.io.envi as envi
import numpy as np
import pickle
import urllib


def envi_to_array(img, hdr):
    """Function which used the spectralpy package to convert envi style images
    to numpy style arrays.

    Parameters:
    -----------
    img - .img file
        .img file with the image

    hdr - header file

    Returns
    -------
    Numpy array - (n by m by d) array

    """
    img = envi.open(hdr, img)

    return img.read_bands(np.arrange(img.shape[2]))

def get_data(dataset='indianpines'):
    """function which extracts hyperspectral images from folder
    or downloads them appropriately.

    Typically this is all free stuff from a known database of free HSI

    Parameters
    ----------
    dataset - str ['indianpines', 'pavia_u', 'pavia_c', 'dc_mall',
                   'salinas', 'salinas_a', 'ksc', 'botswanna', 'cuprite']

    Returns
    -------
    img_data - dictionary { 'original', 'corrected', 'groundtruth'}

    References
    ----------
    Website:
        http://goo.gl/EHZpV1
    """
    #-------------------------------------------------
    # set filename variables for data extraction
    #-------------------------------------------------
    current_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    if dataset in ['indianpines', 'IndianPines', 'IP']:

        # change the working directory

        # check to see if there is a saved file in current directory
        if os.path.exists('raw/hsi/Indian_pines.p'):
            print('-- Indian Pines dataset found locally --')

            # extract data from file
            img_data = pickle.load( open('raw/hsi/Indian_pines.p', 'rb') )
            print('-- Completed --')

        else:
            print('-- Unable to find Indian Pines locally --')
            print('-- trying to download from website --')

            # get the links to the datasets
            fn_original_img = 'http://www.ehu.eus/ccwintco/' + \
                'uploads/2/22/Indian_pines.mat'
            fn_corrected_img = 'http://www.ehu.eus/ccwintco/' + \
                'uploads/6/67/Indian_pines_corrected.mat'
            fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
                'uploads/c/c4/Indian_pines_gt.mat'

            # try to download the files to data folder
            print(fn_gt_img)
            testfile = urllib.URLopener()
            testfile.retrieve(fn_original_img,
                              'raw/hsi/Indian_pines.mat')
            try:
                testfile = urllib.URLopener()
                testfile.retrieve(fn_original_img,
                                  '.raw/hsi/Indian_pines.mat')
                testfile.retrieve(fn_corrected_img,
                                  '../raw/hsi/Indian_pines_corrected.mat')
                testfile.retrieve(fn_gt_img,
                                  '../raw/hsi/Indian_pines_gt.mat')
            except:
                return("-- Unable to download HSIs --")

            # extract the data from file
            raw_data = scipy.io.loadmat(
                'raw/hsi/Indian_pines.mat')
            raw_data_corrected = scipy.io.loadmat(
                'raw/hsi/Indian_pines_corrected.mat')
            raw_data_gt = scipy.io.loadmat(
                'raw/hsi/Indian_pines_gt.mat')

            # set the data to a dictionary of values
            img_data = {}
            img_data['original'] = np.array(
                raw_data['indian_pines']).astype(np.float64)
            img_data['corrected'] = np.array(
                raw_data_corrected['indian_pines_corrected']).astype(np.float64)
            img_data['groundtruth'] = np.array(
                raw_data_gt['indian_pines_gt']).astype(np.int)

            # delete variables to save a bit of space
            del raw_data, raw_data_corrected, raw_data_gt

            # write data to file in folder
            pickle.dump(img_data, open('raw/hsi/Indian_pines.p', 'wb'))

            print('--Completed--')
        os.chdir(current_dir)
        return img_data

# def get_data_generalized(dataset='indianpines'):
#     """function which extracts hyperspectral images from folder
#     or downloads them appropriately.
#
#     Typically this is all free stuff from a known database of free HSI
#
#     Parameters
#     ----------
#     dataset - str ['indianpines', 'pavia_u', 'pavia_c', 'dc_mall',
#                    'salinas', 'salinas_a', 'ksc', 'botswanna', 'cuprite']
#
#     Returns
#     -------
#     img_data - dictionary { 'original', 'corrected', 'groundtruth'}
#
#     References
#     ----------
#     Website:
#         http://goo.gl/EHZpV1
#     """
#     #-------------------------------------------------
#     # set filename variables for data extraction
#     #-------------------------------------------------
#     if dataset in ['indianpines', 'IndianPines', 'IP']:
#
#         fn = 'Indian_pines'
#         # get the links to the datasets
#         fn_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/2/22/Indian_pines_corrected.mat'
#         fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/c/c4/Indian_pines_gt.mat'
#
#     elif dataset in ['pavia_c', 'pavia_center', 'Pavia_C']:
#
#         fn = 'Pavia'
#         # get the links to the datasets
#         fn_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/e/e3/Pavia.mat''
#         fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/5/53/Pavia_gt.mat'
#
#     elif dataset in ['pavia_u', , 'pavia_uni', 'pavia_university','Pavia_U']:
#
#         fn = 'PaviaU'
#         # get the links to the datasets
#         fn_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/e/ee/PaviaU.mat'
#         fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
#             'uploads/5/50/PaviaU_gt.mat'
#
#     elif dataset in ['salinas', 'Salinas']:
#
#         fn = 'salinas'
#         # get the links to the datasets
#         fn_img = 'http://www.ehu.eus/ccwintco/' + \
#                 'uploads/a/a3/Salinas_corrected.mat'
#         fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
#                 'uploads/f/fa/Salinas_gt.mat'
#
#     else:
#         raise ValueError('Need a valid HSI dataset')
#
#     #-------------------------------------------------
#     # Extract data
#     #-------------------------------------------------
#     # check to see if there is a saved file in current directory
#     if os.path.exists('raw/hsi/' + fn + '.p'):
#         print('-- ' + fn + ' dataset found locally --')
#
#         # extract data from file
#         img_data = pickle.load( open('raw/hsi/' + fn + '.p', 'rb') )
#
#     else:
#         print('-- Unable to find ' + fn + ' locally --')
#         print('-- trying to download from website --')
#
#         # try to download the files to data folder
#
#         try:
#             testfile = urllib.URLopener()
#             testfile.retrieve(fn_original_img,
#                               'raw/hsi/' + fn + '.mat')
#             testfile.retrieve(fn_gt_img,
#                               'raw/hsi/' + fn + '_gt.mat')
#         except:
#             exit("-- Unable to download HSIs --")
#
#         # extract the data from file
#         raw_data = scipy.io.loadmat(
#             'raw/hsi/' + fn + '.mat')
#         raw_data_gt = scipy.io.loadmat(
#             'raw/hsi/' + fn + '_gt.mat')
#
#         # set the data to a dictionary of values
#         img_data = {}
#         img_data['original'] = np.array(
#             raw_data[fn]).astype(np.float64)
#         img_data['corrected'] = np.array(
#             raw_data_corrected['indian_pines_corrected']).astype(np.float64)
#         img_data['groundtruth'] = np.array(
#             raw_data_gt['indian_pines_gt']).astype(np.int)
#
#         # delete variables to save a bit of space
#         del raw_data, raw_data_corrected, raw_data_gt
#
#         # write data to file in folder
#         pickle.dump(img_data, open('raw/hsi/indian_pines.p', 'wb'))
#
#         print('--Completed--')
#
#         return img_data
#
#     elif dataset in ['pavia_c', 'pavia_center''Pavia_C']:
#
#         # check to see if there is a saved file in current directory
#         if os.path.exists('raw/hsi/pavia_center.p'):
#             print('-- pavia center dataset found locally --')
#
#             # extract data from file
#             img_data = pickle.load( open('raw/hsi/pavia_center.p', 'rb') )
#
#         else:
#             print('-- Unable to find pavia center locally --')
#             print('-- trying to download from website --')
#
#             # get the links to the datasets
#             fn_original_img = 'http://www.ehu.eus/ccwintco/' + \
#                 'uploads/2/22/Indian_pines.mat'
#             fn_corrected_img = 'http://www.ehu.eus/ccwintco/' + \
#                 'uploads/6/67/Indian_pines_corrected.mat'
#             fn_gt_img = 'http://www.ehu.eus/ccwintco/' + \
#                 'uploads/c/c4/Indian_pines_gt.mat'
#
#             # try to download the files to data folder
#
#             try:
#                 testfile = urllib.URLopener()
#                 testfile.retrieve(fn_original_img,
#                                   'raw/hsi/Indian_pines.mat')
#                 testfile.retrieve(fn_corrected_img,
#                                   'raw/hsi/Indian_pines_corrected.mat')
#                 testfile.retrieve(fn_gt_img,
#                                   'raw/hsi/Indian_pines_gt.mat')
#             except:
#                 exit("-- Unable to download HSIs --")
#
#             # extract the data from file
#             raw_data = scipy.io.loadmat(
#                 'raw/hsi/Indian_pines.mat')
#             raw_data_corrected = scipy.io.loadmat(
#                 'raw/hsi/Indian_pines_corrected.mat')
#             raw_data_gt = scipy.io.loadmat(
#                 'raw/hsi/Indian_pines_gt.mat')
#
#             # set the data to a dictionary of values
#             img_data = {}
#             img_data['original'] = np.array(
#                 raw_data['indian_pines']).astype(np.float64)
#             img_data['corrected'] = np.array(
#                 raw_data_corrected['indian_pines_corrected']).astype(np.float64)
#             img_data['groundtruth'] = np.array(
#                 raw_data_gt['indian_pines_gt']).astype(np.int)
#
#             # delete variables to save a bit of space
#             del raw_data, raw_data_corrected, raw_data_gt
#
#             # write data to file in folder
#             pickle.dump(img_data, open('raw/hsi/indian_pines.p', 'wb'))
#
#             print('--Completed--')
#
#         return img_data
