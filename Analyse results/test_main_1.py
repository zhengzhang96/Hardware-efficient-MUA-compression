# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:10 2021

@author: oscar

No sorting, S = 7, BP = 50, 1 encoder.
"""

# We run it for 10 CVs, for each BP, and for different numbers of S

from functions_1 import *
import numpy as np
import pickle
from scipy.stats import spearmanr
import copy


test_train = 'train'

dictionnary_over_defined_ratio = 1 # How many more symbols to have in the dictionary than in the training data


# Load all test data
if test_train == 'train':
    file_name = 'all_binned_data_end_times_2.pkl'
    
elif test_train == 'test':
    file_name = 'all_binned_TEST_data_end_times.pkl'

with open(file_name, 'rb') as file:      
    results = pickle.load(file)
    all_binned_data = results['all_binned_data']
    bin_vector = results['bin_vector']
    datasets = results['datasets'] 
results = []

bin_resolution = 50
BP_counter = -2
all_data = all_binned_data[BP_counter]

S = 3
max_symbol = int(S-1) # Load max_symbol for each BP


encoder = ['0', '10', '110', '1110', '11110', '111111', '111110']
#CLV = [1,2,3,4,5,6,6]
CLV = [1,2,2]

# Collate all channels
BR = []
stored_plots = []
stored_hist = []
mean_hist = []
collated_data = []
stored_hist_skew = []
for dataset_count, data in enumerate(all_data):
    collated_data = data
    nb_channels = len(collated_data)
    
        
    print('BP: ' + str(bin_resolution) + '; S: ' + str(int(S)))
    
    
    # =============================================================================
    # # Load all possible LUTs
    # file_name = 'Stored_LUTs_S_'+str(int(S))+'_final.pkl'
    # with open(file_name, 'rb') as file:      
    #     CLVs = pickle.load(file)
    #     nb_CLVs = len(CLVs)
    #     CLVs = np.array(CLVs,dtype=object)
    # 
    # if len(CLVs[0,:]) != max_symbol + 1:
    #     input('We have a problem, S not equal to CLV length')
    # 
    # =============================================================================
    
    # Get training data histogram (floor histogram to previous power of 2)
    histograms = np.zeros((max_symbol+1,nb_channels))
    len_data = np.zeros((nb_channels,1))
    symbol_list_bin_limits = np.arange(-0.5,max_symbol+1.5,1)#np.unique(MUA_binned_validation)
    hist_skew = []
    for chan_index, train_data in enumerate(collated_data):
        
        # Clip at S
        train_data[train_data> max_symbol] = max_symbol
        
        # Just histogram, no sorting
        temp_hist_train = np.histogram(train_data, symbol_list_bin_limits)
        histograms[:,chan_index] = temp_hist_train[0]
        len_data[chan_index] = np.sum(temp_hist_train[0])
        hist_skew.append(skew(temp_hist_train[0]))
        #BR_chan.append((np.sum(CLVs[LUT_index,:] * histograms[:,chan_index])))
        
        # Store time domain plots
        if chan_index < 5:
            stored_plots.append(train_data)
    
    mean_hist = np.zeros((len(histograms[:,0]),len(histograms[0,:])))
    for i in np.arange(len(histograms[0,:])):
        mean_hist[:,i] = histograms[:,i] / np.sum(histograms[:,i])
    
    stored_hist.append(mean_hist)
    stored_hist_skew.append(hist_skew)  
    # Get dot product of CLVs and training data histogram, proxy for CR
    dot_prod = np.matmul(np.transpose(histograms),np.transpose(CLV))
    average_bits_per_sample = np.zeros((nb_channels))
    for i in np.arange(nb_channels):
        average_bits_per_sample[i] = dot_prod[i] / len_data[i]
    
    BR.append(np.mean(average_bits_per_sample ) / (bin_resolution/1000))
    #input('ye')
    #mean_hist.append(np.nanmean(histograms/np.nansum(histograms,axis=1),axis=1))

low_est_CR, upper_est_CR, compared_to_1_ms_bins = compare_CR_to_uncompressed(bin_resolution,max_symbol,np.mean(dot_prod / len_data),dictionnary_over_defined_ratio)

print('BR results (Flint, Sabes, Brochier): ', BR)
print('Total power per channel: ', 0.96 + np.array(BR)*0.02)
# =============================================================================
# 
# if test_train == 'train':
#     file_name = 'stored_hist_skew_train.pkl'
#     with open(file_name, 'wb') as file:
#           
#         results = {'stored_hist_skew_train': stored_hist_skew,
#                    'stored_histograms_train':stored_hist}
#         # A new file will be created
#         pickle.dump(results, file)
#         
# elif test_train == 'test':
#     file_name = 'stored_hist_skew_test.pkl'
#     with open(file_name, 'wb') as file:
#           
#         results = {'stored_hist_skew_test': stored_hist_skew,
#                    'stored_histograms_test':stored_hist}
#         # A new file will be created
#         pickle.dump(results, file)
#         
#                
# 
# =============================================================================
