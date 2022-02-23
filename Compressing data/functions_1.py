# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:00:55 2021

@author: oscar
"""

import numpy as np
import math

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
