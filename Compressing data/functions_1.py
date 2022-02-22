# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:00:55 2021

@author: oscar
"""

import numpy as np
import math
from scipy.io import loadmat

def load_Flint_data(rec,subject):

    # Load MUA Flint data, (load from .mat files)
    #mua_flint_directory = "D:\\Dropbox (Imperial NGNI)\\NGNI Share\\Workspace\\Oscar\\Work\\MUA compression\\Flint_data\\raw_data"
    mua_flint_directory = 'XXX'
    
    if mua_flint_directory == 'XXX':
        print('Fill in Sabes directory path in function\n')
        return None
    
    file_path = mua_flint_directory + "\\Rec_" + str(rec) + "_Subject_" + str(subject) + ".mat"
    print('Loading data: Recording ' + str(rec) + '; Subject: ' + str(subject))
    data = loadmat(file_path)
    return data

def load_Thomas_Brochier_data(monkey):
    
    #Brochier_directory = 'D:\\Dropbox (Imperial NGNI)\\NGNI Share\\Workspace\\Oscar\\Work\\MUA compression\\Thomas_Brochier_data'
    Brochier_directory = 'XXX'
    
    if Brochier_directory == 'XXX':
        print('Fill in Brochier directory path in function\n')
        return None
    else:
        if monkey == 'N':
            file_name = 'Monkey_N_BP_1ms';
        elif monkey == 'L_noisy':
            file_name = 'Monkey_L_noisy_BP_1ms';
        elif monkey == 'L_clean':
            file_name = 'Monkey_L_clean_BP_1ms';
        else:
            print('Enter valid subject title (M or L)')
        file_path = Brochier_directory + '\\' + file_name
        data = loadmat(file_path)
        return data


def load_Sabes_lab_data(monkey,file_nb):
    
    #Sabes_directory = 'D:\\Dropbox (Imperial NGNI)\\NGNI Share\\Workspace\\Oscar\\Work\\MUA compression\\Sabes_lab_data'
    Sabes_directory = 'XXX'
    
    if Sabes_directory == 'XXX':
        print('Fill in Sabes directory path in function\n')
        return None
    else:
        
        if monkey == 'Indy':
            file_name = Sabes_directory + '\\filenames_Indy.txt'
        elif monkey == 'Loco':
            file_name = 'missing'
        else:
            print('Enter valid subject title (Indy or Loco)')
            
        # Look at all stored file names, we have it this way so the indexing is 
        # consistent, always relative to the .txt file 
        with open(file_name) as f:
            lines = f.readlines()
        
        file = lines[file_nb].replace('\n','.mat')
        file_path = Sabes_directory + '\\' + file
        data = loadmat(file_path)
        return data

def bin_MUA_data(MUA,bin_res):
    counter = 0
    binned_MUA = np.zeros([math.ceil(len(MUA[:,1])/bin_res),len(MUA[1,:])])
    for bin in range(math.ceil(len(MUA[:,1])/bin_res)):
        if bin != math.ceil(len(MUA[:,1])/bin_res):
            temp = np.sum(MUA[counter:counter+bin_res,:],0)
        else:
            temp = np.sum(MUA[counter:len(MUA[:,1]),:],0)
        
        binned_MUA[bin,:] = temp
        counter = counter + bin_res
        
    binned_MUA = binned_MUA.astype(int)
    return binned_MUA


def online_histogram_w_sat_based_nb_of_samples(data_in,sample_val_cutoff, max_firing_rate):

    # We consider the histogram to be full when "sample_val_cutoff" values have 
    # been entered into it.
    # Inputs:
    # data_in = 1d vector of MUA data from 1 channel.
    # sample_val_cutoff = how mnay values the histogram will measure until we 
    # consider the histogram training period to have ended.
    # max_firing_rate: S-1, max value that we consider in the MUA data.
    # Outputs:
    # approx sorted histogram, how many samples we measure (just for testing purposes)

    hist = {'0':0}
    flag_1 = False
    i = 0
    while not flag_1: # the histogram isn't full yet
    
        # Saturate the histogram at the max firing rate
        if data_in[i] >= max_firing_rate:
            data_in[i] = max_firing_rate
    
        symbol = str(data_in[i])
        
        if symbol in hist: # If this symbol is represented in the histogram
            hist[symbol] += 1
        else: # If this symbol is new in the histogram
            hist[symbol] = 1
            
        # If the histogram is full, end the while loop
        hist_count = 0
        for symbol_hist in hist:
            hist_count += int(hist.get(str(symbol_hist)))
        if hist_count > sample_val_cutoff-1: 
            flag_1 = True
            
        # If we've exceeded the number of samples in the data, end the while loop
        if i+1 == len(data_in): 
            flag_1 = True
                
        i += 1 # Increment counter
        
    return hist, i


# Approx sort used in the work, where the histogram is assumed to follow a 
# unimodal distribution. The peak in the histogram is identified and given an
# index of 0, and values on either side are iteratively assigned the next 
# indices.
def approx_sort(hist):
    idx = np.arange(0,len(hist))
    p_idx = np.argmax(hist)
    if (p_idx>len(hist)/2): # peak shows on right half
        right = np.arange(2,(len(hist)-1-p_idx)*2+1,2) #idx on the right (even or odd doesn't matter)                 
        idx = np.delete(idx,right) # remove used idx
        left = idx
    else:                   # peak shows on left half
        left = np.arange(1,(2*p_idx-1)+1,2)
        idx = np.delete(idx,left)
        right = idx
    
    idx = np.hstack((np.flip(left),right))
    idx = np.argsort(idx)

    return idx.astype(int), hist[idx.astype(int)]
