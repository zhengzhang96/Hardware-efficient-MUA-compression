# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:10 2021

@author: oscar

Full system, S = 3, BP = 50, 1 encoder.
"""

from functions_1 import *
import numpy as np
import pickle
import re


################### SELECT OPTION, PARAMETERS ############################

# Parameters
datasets =['Flint','Sabes']
train_or_test = 'test' # 'train' for training data (A), 'test' for testing data (B)

bin_resolution = 50
BP_counter = -2 # Gives BP of 50 ms, how we index the 50 ms BP data from all_binned_data_XXX.pkl
S = 3
hist_memory = 6 # bits
encoder = ['0', '10', '11']
SCLV = [1,2,2]

# Specify root directory (where directories.txt file is located)
root_directory = r'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload code'

##########################################################################

# Read directories.txt file
with open(root_directory + '\\directories.txt') as f:
    lines = f.readlines()

# Get path to Formatted data
for path in lines:
    if path.startswith('Formatted_data_path'):
        pattern = "'(.*?)'"
        data_path = re.search(pattern, path).group(1)

# Load all test data
file_name = data_path + '\\all_binned_data_' +train_or_test+'.pkl'


with open(file_name, 'rb') as file:      
    results = pickle.load(file)
    all_binned_data = results['all_binned_data']
    bin_vector = results['bin_vector']
    datasets = results['datasets'] 
results = []

all_data = all_binned_data[BP_counter]
sample_val_cutoff = pow(2,hist_memory)
max_symbol = int(S-1) # Load max_symbol for each BP


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
    
    # Get data histograms
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
        sort_indices, sort_hist = approx_sort(temp_hist_val)
        val_histograms[:,channel] = sort_hist
                
        # Get validation histogram after assignment (or in this case mapping),
        # used to measure BR
        end_cutoff[channel] = int(sample_counter_cutoff[channel]) + int(len(implant_data)/2)
        
        # We need to ensure that the sorting of the histogram is done
        # according to the mapping
        temp_hist_val_post = np.histogram(implant_data[int(sample_counter_cutoff[channel]):int(end_cutoff[channel])], symbol_list_bin_limits)[0]
        val_histograms_post[:,channel] = [temp_hist_val_post[i] for i in sort_indices]
                
        len_data[channel] = np.sum(temp_hist_val_post)
    
        # Store time domain plots
        if channel < 5:
            stored_plots.append(implant_data)
            

    # Get BR of mapped histograms
    mean_hist = np.zeros((len(val_histograms_post[:,0]),len(val_histograms_post[0,:])))
    for i in np.arange(len(val_histograms_post[0,:])):
        mean_hist[:,i] = val_histograms_post[:,i] / np.sum(val_histograms_post[:,i])
    stored_hist.append(mean_hist) 
    
    # Get dot product of SCLVs and data histogram, to get BR
    dot_prod = np.matmul(np.transpose(val_histograms_post),np.transpose(SCLV))
    average_bits_per_sample = np.zeros((nb_channels))
    for i in np.arange(nb_channels):
        average_bits_per_sample[i] = dot_prod[i] / len_data[i]
    
    BR.append(np.mean(average_bits_per_sample ) / (bin_resolution/1000))

if len(BR) == 2:
    BR.append(float('nan'))

print('BR results for ' + train_or_test +' data (Flint, Sabes, Brochier): ', BR)
print('Total power per channel: ', 0.96 + np.array(BR)*0.02)
