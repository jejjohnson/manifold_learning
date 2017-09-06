# -*- coding: utf-8 -*-
"""
Date Created: Saturday, 11th February, 2017
Author: J. Emmanuel Johnson

"""
# import appropriate packages
import urllib.request
import scipy.io as sio
import pickle
import os
import logging

# configure logging
logging.basicConfig(filename='ip_logs.log', level=logging.DEBUG)


def get_indianpines(verbose=None):
    """Imports Indian Pines dataset   
    Information
    -----------
    Author : J Emmanuel Johnson
    References
    ----------
    chris' Sandbox for importing data
        https://goo.gl/5dYjdj
    """
    # URLS of the sample dataset
    ip_urls = list()
    ip_urls.append("http://www.ehu.eus/ccwintco/uploads/2/22/" 
                   "Indian_pines.mat")
    ip_urls.append("http://www.ehu.eus/ccwintco/uploads/6/67/" 
                   "Indian_pines_corrected.mat")
    ip_urls.append("http://www.ehu.eus/ccwintco/uploads/c/c4/"
                   "Indian_pines_gt.mat")
    # names of datases
    ip_names = list()
    ip_names.append('Indian_pines.mat')
    ip_names.append('Indian_pines_corrected.mat')
    ip_names.append('Indian_pines_gt.mat')
    # keys of datasets
    data_names = list()
    data_names.append('indian_pines')
    data_names.append('indian_pines_corrected')
    data_names.append('indian_pines_gt')
    
    print('Completed url saving.')
    
    # check if pickle file is there
    if os.path.exists("indianpines.pickle"):
        # log success
        logging.info("Found indianpines.pickle file.")
        
        # print success if verbose
        if verbose:
            print("--IndianPines file found locally.")
            
        # check if the files are the same
        with open('indianpines.pickle', 'rb') as handle:
            hsi_data = pickle.load(handle)
        
    else:        
        # log failure 
        logging.info("Unable to find file locally.")
        
        # print failure if verbose
        if verbose:
            print("--trying to download from server.")
            
        # try to acquire
        try:
            print("--- trying to download with urllib2")

            for (url, name) in zip(ip_urls, ip_names):
                f = urllib.request.urlopen(url)
                data = f.read()
                with open(name, "wb") as code:
                    code.write(data)
        
        except:
            exit("--- unable to download with urllib2")
            
        # create an empty dictionary to hold the data 
        hsi_data = {}
        
        # loop through key and filename
        for name, file in zip(data_names, ip_names):
            
            # import .mat file
            temp = sio.loadmat(file)
            
            # save hsi data in dictionary
            hsi_data[name] = temp[name]
        
        # Save data in pickle file
        with open('indianpines.pickle', 'wb') as handle:
            pickle.dump(hsi_data, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)
            
        logging.info("Saved data to pickle file.")
        
    return hsi_data

    
if __name__ == "__main__":
    # test
    hsi_data = get_indianpines(verbose=False)
