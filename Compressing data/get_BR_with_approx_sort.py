# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:10 2021

@author: oscar

Script for training-validation process, getting BRs for each BP, S, number of 
encoders and histogram size combination. We use an approx. sort algorithm , detialed
in the associated article.
"""

from functions_1 import *
import numpy as np
import pickle
import copy
import re


################### SELECT OPTION, PARAMETERS ############################

# Parameters
samples_per_channel_for_histogram_vector = pow(2,np.array([2,3,4,5,6,7,8,9,10])) # size of on-implant histogram
train_percentage = 50 # half of channels are for training, half for validation, for each cross-validation (CV) split
how_many_channels_Sabes = 2000 # we limit the number of Sabes channels since there ar eonly 960 Flint channels, this prevents the results from overfitting to the Sabes data
nb_CV_iterations = 30

# Specify root directory (where directories.txt file is located)
root_directory = r'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload code'

##########################################################################


# Read directories.txt file
with open(root_directory + '\directories.txt') as f:
    lines = f.readlines()

# Get path to Formatted data
for path in lines:
    if path.startswith('Formatted_data_path'):
        pattern = "'(.*?)'"
        data_directory = re.search(pattern, path).group(1)
        
# Get results directory
for path in lines:
    if path.startswith('BR_approx_sort_results'):
        pattern = "'(.*?)'"
        results_directory = re.search(pattern, path).group(1)


# Get SCLV directory
for path in lines:
    if path.startswith('SCLV_path'):
        pattern = "'(.*?)'"
        SCLV_directory = re.search(pattern, path).group(1)
        
    


# Load binned MUA data
file_name = data_directory + '\\all_binned_data_train.pkl'
with open(file_name, 'rb') as file:      
    results = pickle.load(file)
    all_binned_data = results['all_binned_data']
    bin_vector = results['bin_vector']
    datasets = results['datasets'] 
results = [] # clear variable


# Iterate through cross-validation iterations
for CV_iteration in np.arange(1,nb_CV_iterations,1):

    # Iterate through different BPs
    for BP_counter, bin_resolution in enumerate(bin_vector):

        all_data = all_binned_data[BP_counter] # index the data binned at the desired BP
        
        # Split channels into train and test, with shuffling
        MUA_binned_train = []
        MUA_binned_validation = []
        
        # Iterate through Flint and Sabes datasets, stored separately in 'all_data' variable
        for dataset_count, data in enumerate(all_data):
            
            # Shuffle channels
            shuffled_channels = np.random.permutation(len(data))
            data = [data[i] for i in shuffled_channels]
            
            # Limit the Sabes data number of channels
            if dataset_count == 1: # Sabes
                data = data[:how_many_channels_Sabes]
                
            # Index at which below are train channels and above are validation channels
            train_cutoff = int(np.round(train_percentage * len(data) / 100))
            
            # Assign train and validation channels
            MUA_binned_train.extend(data[:train_cutoff])
            MUA_binned_validation.extend(data[train_cutoff:])
               
        
        nb_train_channels = len(MUA_binned_train)
        nb_val_channels = len(MUA_binned_validation)
        nb_channels = nb_train_channels + nb_val_channels
        all_data = []
        data = []
        
        # Iterate through S values
        for S in np.arange(2,11,1):
            
            # Work on copies of the data, keeps the starting point the same for each S
            MUA_binned_train_copy = copy.deepcopy(MUA_binned_train)
            MUA_binned_validation_copy = copy.deepcopy(MUA_binned_validation)

            print('BP: ' + str(bin_resolution) + '; S: ' + str(int(S)))
 
            # Set the max firing rate for each BP, according to S
            max_firing_rate = int(S-1)
            
            # Load all possible SCLVs
            # NOTE: if these 
            file_name = 'Stored_SCLVs_S_'+str(int(S))+'.pkl'
            try:
                with open(SCLV_directory + '\\' + file_name, 'rb') as file:      
                    SCLVs = pickle.load(file)
                    nb_SCLVs = len(SCLVs)
                    SCLVs = np.array(SCLVs,dtype=object)
            except:
                print('If SCLV files have not been produced, use "produce_all_SCLVs_given_S" script to produce them.')
            
            if len(SCLVs[0,:]) != max_firing_rate + 1:
                input('We have a problem, S not equal to SCLV length')
            
            
            # Get training data histogram for each channel
            # Note: We only work on the histograms:
            # it saves time in that it's much more computationally efficient to 
            # do the sorted histogram and SCLV dot product than encode the 
            # data and check it's length.
            histograms = np.zeros((max_firing_rate+1,nb_train_channels))
            symbol_list_bin_limits = np.arange(-0.5,max_firing_rate+1.5,1)
            for chan_index, train_data in enumerate(MUA_binned_train_copy): # iterate through training data channels
                
                # Saturate dynamic range at S
                train_data[train_data> max_firing_rate] = max_firing_rate
                
                # Get histogram
                temp_hist_train = np.histogram(train_data, symbol_list_bin_limits)
                histograms[:,chan_index] = np.flip(np.sort(temp_hist_train[0]))
            
                        
            # Validation data histograms
            val_histograms_memory = []
            val_histograms_post_memory = []
            sample_counter_cutoff = np.zeros((nb_val_channels,len(samples_per_channel_for_histogram_vector))) # cutoff index after which we use the validation data for measuring compression (before cutoff is for assignment)
            end_cutoff = np.zeros((nb_val_channels,len(samples_per_channel_for_histogram_vector))) # cutoff index after which the validation data is no longer used
            
            # Get val data histogram, depending on hist memory size
            for hist_counter, sample_val_cutoff in enumerate(samples_per_channel_for_histogram_vector): # iterate through histogram sizes
                val_histograms = np.zeros((max_firing_rate+1,nb_val_channels))
                val_histograms_post = np.zeros((max_firing_rate+1,nb_val_channels))
                skipped_val = 0 # used to test the code
                for channel, val_data in enumerate(MUA_binned_validation_copy): # iterate through validation data channels
                    
                    # Saturate dynamic range at S
                    val_data[val_data> max_firing_rate] = max_firing_rate
                    
                    # Get sample cutoff for the given histogram size
                    temp_dict, sample_counter_cutoff[channel,hist_counter] = online_histogram_w_sat_based_nb_of_samples(val_data,sample_val_cutoff,max_firing_rate)
                    
                    # Get validation histogram for assignment, counted up to when 
                    # histogram is saturated
                    temp_hist_val = np.histogram(val_data[:int(sample_counter_cutoff[channel,hist_counter])], symbol_list_bin_limits)[0]
                    
                    # KEY FEATURE: No sorting of validation data in this version
                    # Use zheng sort
                    approx_sort_indices, approx_sort_hist = approx_sort(temp_hist_val)
                    val_histograms[:,channel] = approx_sort_hist
             
                    # Get validation histogram after assignment (all the data used to
                    # measure BR)
                    end_cutoff[channel,hist_counter] = int(sample_counter_cutoff[channel,hist_counter]) + int(len(val_data)/2)
                    
                    # If not enough data (if more than half the data was used for assignment), we skip as we want eahc histogrma size to use the same amount of samples for compression (shows up as NaN in the BR)
                    if end_cutoff[channel,hist_counter] > len(val_data):
                        skipped_val += 1
                        continue
                    
                        
                    # Get the histogram of the to-be-compressed validation data (post-assignment data)
                    temp_hist_val_post = np.histogram(val_data[int(sample_counter_cutoff[channel,hist_counter]):int(end_cutoff[channel,hist_counter])], symbol_list_bin_limits)[0]
                    
                    # We need to ensure that the sorting of the histogram is done
                    # according to the sorting given during assignment
                    val_histograms_post[:,channel] = [temp_hist_val_post[i] for i in approx_sort_indices]
                    

                    
                # Code Test: make sure nb of columns of zeros in val_histograms_post
                # matches the amount of skipped histograms
                temp_test = np.sum(np.sum(val_histograms_post,axis=0)==0)
                if temp_test != skipped_val:
                    try:
                        raise ValueError("Check code: the amount of skipped histograms should match the number of columns of zeros in the post-assignment validation histograms")
                        raise Exception('This is the exception you expect to handle')
                    except Exception as error:
                        print('Caught this error: ' + repr(error))
                    
            
                # Store validation assignment and post-assignment histograms, one of each for each histogram size
                val_histograms_memory.append(val_histograms)
                val_histograms_post_memory.append(val_histograms_post)
            
  
            # Get the percentage of post- to pre-assignement val data, just so we know how much of the data wa sused for compression, relatively
            stored_val_BR_data_proportion = (end_cutoff.astype(int) - sample_counter_cutoff.astype(int)) / end_cutoff.astype(int)
            
            # Iterate through rounds of training, reduce the number of SCLVs as we go by removing
            # the least popular ones. With each round, we find the subset of most popular SCLVs.
            stored_SCLVs = []
            stored_all_var_BRs = [] # Store CRs
            stored_hist_SCLVs = [] # to see which channels were assigned to which SCLVs
            counter_round = 0
            SCLV_index = np.zeros((nb_train_channels,nb_SCLVs))
            while nb_SCLVs != 0: # while there is an SCLV left
                all_var_BRs = []
                
                # Store the CLVs in this round (each round has a reduced amount, we remove
                # the least popular ones)
                stored_SCLVs.append(SCLVs)
                
                # Get dot product of CLVs and training data histogram, proxy for CR
                dot_prod =  np.matmul(np.transpose(histograms),np.transpose(SCLVs))
                
                # Assign channels to SCLVs based on minimum dot product value
                # (Smallest dot product = smallest codeblock = biggest CR)
                for channel in np.arange(nb_train_channels):
                    SCLV_index[channel,counter_round] = np.argmin(dot_prod[channel,:]) # which SCLV for given channel
                
                # Observe SCLV-assignment histogram
                SCLV_hist_bin_limits = np.arange(0,nb_SCLVs+1)-0.5#np.unique(MUA_binned_validation)
                hist_SCLV = np.histogram(SCLV_index[:,counter_round], SCLV_hist_bin_limits)[0] # How many channels for each SCLV
                
                stored_hist_SCLVs.append(hist_SCLV) # store histogram of channels to SCLVs for this round
                
                #input('val enter')
                # At this point, we know which channel is assigned to which SCLV.
                # Next, we need to figure out what the CR will be for each channel, using
                # the rest of the validation data.
                
                # Get CR on validation data (post assignment), with different histogram memory pool sizes
                for hist_counter, val_histogram in enumerate(val_histograms_memory):
                    # Get dot product of SCLVs and histogram, during assignment. Gives BR
                    # used during assignment -> we assign channel to SCLV with the smallest 
                    # BR, given the assignement validation data histogram
                    val_dot_prod =  np.matmul(np.transpose(val_histogram),np.transpose(SCLVs))
                    
                    # Validation data used post assignment, used to determine actual CR
                    # post-assignment
                    val_histogram_post = val_histograms_post_memory[hist_counter]
            
                    # Test: the sum of elements in val_histogram_post should be the
                    # same for each hist_mem size, given that half of the data is used
                    for hist_counter_temp in np.arange(len(val_histograms_post_memory)):
                        if hist_counter_temp == 0:
                            test_val = np.sum(val_histograms_post_memory[hist_counter])
                        if np.sum(val_histograms_post_memory[hist_counter]) != test_val:
                            try:
                                raise ValueError("Check code: sum of elements in val_histogram_post should be the same for each hist_mem size")
                                raise Exception('This is the exception you expect to handle')
                            except Exception as error:
                                print('Caught this error: ' + repr(error))
                            
                    # Here we measure the BRs from the validation data after 
                    # assignment to their ideal SCLVs, based on the sample 
                    # histogram. We do this indirectly by measuring the CR relative
                    # to 1 ms BP, S = 2 (i.e. 1000 bps), and later derive the actual BR.
                    val_BRs = []
                    for channel in range(nb_val_channels): # iterate through validation channels
                        
                        # Assign SCLV with max projected CR (according to validation data
                        # in assignement block) to channel
                        encoder_index = np.argmin(val_dot_prod[channel,:])
                        
                        # Get data length of val data (post assignment)
                        val_data_len = np.sum(val_histogram_post[:,channel])
                        
                        # Get CR of assigned SCLV
                        # Note: If val_histogram_post is a vector of 0s (since over half the recording was used for assignment)
                        # this returns NaN 
                        average_bits_per_symbol = (np.sum(SCLVs[encoder_index,:] * val_histogram_post[:,channel]))/val_data_len
                        
                        # Compression ratio relative to 1 ms BP, from this we can derive the exact BR
                        BR = 1000/(bin_resolution/average_bits_per_symbol)
                        val_BRs.append(BR)
                        
                        
                    all_var_BRs.append(val_BRs)
                    
                    
                print('CV round: '+str(CV_iteration) + '; SCLV reduction counter: ' + str(counter_round))

                # Store CRs. Each sub-element corresponds to the results for 1 
                # SCLV reduction round.
                # The next layer corresponds to the different memory sizes.
                # The last layer contains the individual CRs for each channel.
                stored_all_var_BRs.append(all_var_BRs)

                
                # Calculate the mean training channel CR after the removal of each SCLV. Remove
                # the SCLV whose removal led to smallest CR (the least useful SCLV).
                if nb_SCLVs != 1:
                    training_SCLV_rem_CR = np.zeros((nb_SCLVs))
                    for SCLV_removal in np.arange(nb_SCLVs):
                        dot_prod_removed_SCLV = np.delete(dot_prod,SCLV_removal,axis=1)
                        training_SCLV_rem_CR[SCLV_removal] = np.mean(np.min(dot_prod_removed_SCLV,axis=1))
                    
                    SCLVs = np.delete(SCLVs,np.argmin(training_SCLV_rem_CR), axis=0)  
                else: # last round
                    SCLVs = np.delete(SCLVs,0, axis=0) 
             
                nb_SCLVs = len(SCLVs)
                counter_round += 1

                # Ends when all SCLVs have been removed
            
            
            ## Save results
            file_name = results_directory + '\\BRs_S_'+str(int(S))+'_BP_' + str(bin_resolution) +'_CV_' + str(CV_iteration) + '.pkl'
            with open(file_name, 'wb') as file: 
                results = {'stored_all_var_BRs': stored_all_var_BRs,
                           'stored_SCLVs': stored_SCLVs,
                           'stored_hist_SCLVs': stored_hist_SCLVs,
                           'stored_val_BR_data_proportion': stored_val_BR_data_proportion} # Shows how much of the validation data is used for assignment vs CR
                # A new file will be created
                pickle.dump(results, file)
