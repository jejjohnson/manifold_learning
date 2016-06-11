# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:41:29 2016

@author: eman
"""
import os
import scipy.io
import spectral.io.envi as envi
import numpy as np
import pickle

def get_hsi(datatype='vcu', system='linux'):
    

        
    
    # defining which system I am currently in
    if system in ['linux']:
        
        path = "/media/eman/STORAGE_DRIVE/Documents/code/" + \
               "coding_projects/data/hsi/vcu/"
    else:
        raise ValueError('Sorry, need a system data path.')
    
    try:
        return load_hsi(system=system)
    except:
        print('No hsi data found. Loading data...')
    # defining which dataset I would like to acquire
    if datatype in ['vcu', 'VCU']:
        imga = '400m_subseta.img'               # subset image a
        imgb = '2000m_subseta.img'              # subset image b
        hdra = '400m_subseta.hdr'               # subset image a hdr
        hdrb = '2000m_subseta.hdr'              # subset image b hdr
        imggta = '400m_classmap.img'             # subset image gt a
        imggtb = '2000m_classmap.img'            # subset image gt b
        hdrgta = '400m_classmap.hdr'             # subset image gt hdr a
        hdrgtb = '2000m_classmap.hdr'            # subset image gt hdr b
        
    else:
        raise ValueError('Sorry, Need a dataset.')
        
    # changing the paths
    current_path = os.getcwd()
    os.chdir(path)

    
    # get data
    img1 = envi_to_array(imga, hdra)
    img2 = envi_to_array(imgb, hdrb)
    gt1 = envi_to_array(imggta, hdrgta)
    gt2 = envi_to_array(imggtb, hdrgtb)
    
    img_data = {'img1': img1, 'img2': img2, 'gt1': gt1, 'gt2': gt2}
    
    pickle.dump(img_data, open('save.p', 'w'))

    
    # change back to the current path
    os.chdir(current_path)
    
    return img1, img2, gt1, gt2


    

def load_hsi(system='linux'):  
    
    if system in ['linux']:
        path = "/media/eman/STORAGE_DRIVE/Documents/code/" + \
               "coding_projects/data/hsi/vcu/"
    else:
        raise ValueError('Sorry, need a system data path.')
    
    # changing the paths
    current_path = os.getcwd()
    os.chdir(path)   
    
    img_data = pickle.load( open('save.p', 'r'))
    
    # change back to the current path
    os.chdir(current_path)
    
    return img_data

def envi_to_array(img, hdr):
    
    img = envi.open(hdr, img)
    
    return img.read_bands(np.arange(img.shape[2]))
    
    