# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:10 2021

@author: oscar

Full system, S = 3, BP = 50, 1 encoder.
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
hist_memory = 6 # bits
sample_val_cutoff = pow(2,hist_memory)
max_symbol = int(S-1) # Load max_symbol for each BP


encoder = ['0', '10', '11']
CLV = [1,2,2]

# Collate all channels
BR = []
stored_plots = []
stored_hist = []
mean_hist = []
collated_data = []
for dataset_count, data in enumerate(all_data):
    collated_data = data
    nb_channels = len(collated_data)
    
        
    print('BP: ' + str(bin_resolution) + '; S: ' + str(int(S)))
    
    
    # Get training data histogram (floor histogram to previous power of 2)
    val_histograms = np.zeros((max_symbol+1,nb_channels))
    val_histograms_post = np.zeros((max_symbol+1,nb_channels))
    sample_counter_cutoff = np.zeros((nb_channels))
    end_cutoff = np.zeros((nb_channels))

    len_data = np.zeros((nb_channels,1))
    symbol_list_bin_limits = np.arange(-0.5,max_symbol+1.5,1)#np.unique(MUA_binned_validation)
    for channel, implant_data in enumerate(collated_data):
        
        # Clip at S
        implant_data[implant_data> max_symbol] = max_symbol
               
        # Get val histogram
        # Get sample cutoff for the given histogram size
        temp_dict, sample_counter_cutoff[channel] = online_histogram_w_sat_based_nb_of_samples(implant_data,sample_val_cutoff,max_symbol)

        # Get validation histogram for assignment, counted up to when 
        # histogram is saturated
        temp_hist_val = np.histogram(implant_data[:int(sample_counter_cutoff[channel])], symbol_list_bin_limits)[0]
             
        # Use zheng sort
        zheng_indices, zheng_sort_hist = zheng_algo_sort(temp_hist_val)
        val_histograms[:,channel] = zheng_sort_hist
        # NOTE: NO NEED FOR VAL HISTOGRAM, ONLY FOR THE MAPPING.
                
        # Get validation histogram after assignment (all the data used to
        # measure CR)
        end_cutoff[channel] = int(sample_counter_cutoff[channel]) + int(len(implant_data)/2)
        
        # We need to ensure that the sorting of the histogram is done
        # according to the sorting given during assignment
        temp_hist_val_post = np.histogram(implant_data[int(sample_counter_cutoff[channel]):int(end_cutoff[channel])], symbol_list_bin_limits)[0]
        val_histograms_post[:,channel] = [temp_hist_val_post[i] for i in zheng_indices]
                
        len_data[channel] = np.sum(temp_hist_val_post)
    
# =============================================================================
#         # Just histogram, no sorting
#         temp_hist_train = np.histogram(train_data, symbol_list_bin_limits)
#         histograms[:,chan_index] = temp_hist_train[0]
#         len_data[chan_index] = np.sum(temp_hist_train[0])
#         hist_skew.append(skew(temp_hist_train[0]))
#         #BR_chan.append((np.sum(CLVs[LUT_index,:] * histograms[:,chan_index])))
# =============================================================================

        # Store time domain plots
        if channel < 5:
            stored_plots.append(implant_data)
            

    # Get BR of mapped histograms
    
    mean_hist = np.zeros((len(val_histograms_post[:,0]),len(val_histograms_post[0,:])))
    for i in np.arange(len(val_histograms_post[0,:])):
        mean_hist[:,i] = val_histograms_post[:,i] / np.sum(val_histograms_post[:,i])
    stored_hist.append(mean_hist) 
    
    # Get dot product of CLVs and training data histogram, proxy for CR
    dot_prod = np.matmul(np.transpose(val_histograms_post),np.transpose(CLV))
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
