# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 03:53:30 2022

Basically, we're checking how many channels we can have on a FPGA of a 
certain size. Some channels have higher BRs that others, and so the 
communication power, and therefore heat flux, varies. So what's the worst 
case? We find it by taking a random sample of all considered channels, taking 
the total BR, and adding the ADC and static power the comm energy per bit to
get the total power. If it exceeds the heating limit, we mark it (matrix
called "nb_channels_that_exceeded_power_budget" at the end). Each column 
in "nb_channels_that_exceeded_power_budget" corresponds to a number of 
considered channels, given in 'nb_channels_vec'. This gives a 
permutation-derived p-value as to how likely having a number of channels is 
to exceed the power budget.

@author: oscar
"""
import pickle
import numpy as np
import re
#from random import sample

################### SELECT OPTION, PARAMETERS ############################

# Parameters        
nb_channels_vec = np.arange(290, 320) # Scanning a useful range of Z values (nb of channels) to see when heat budget is exceeded
nb_random_CVs = 100000
static_process_power = 0.1618e-3  # FPGA static power in W
chan_processing_power = 0.96e-6 # FPGA processing power per channel power in W
comm_energy = 20e-9 # in J/bit
ADC_power = 0 # count it as off-FPGA
#total_power_budget = 10e-3 * (1.4e-1 * 1.48e-1) # for 1.48 mm x 1.48 mm scale FPGA
total_power_budget = 10e-3 * (2.5e-1 * 2.5e-1) # for 2.5 mm x 2.5 mm scale FPGA


# Specify root directory (where directories.txt file is located)
root_directory = r'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload code'

# Architecture, no-sort or approx-sort, we selected parameters
# for both that seem to offer good overall system performance
approx_or_no_sort = 'approx-sort' 

##########################################################################

# Read directories.txt file
with open(root_directory + '\\directories.txt') as f:
    lines = f.readlines()

# Get BR results directory
if approx_or_no_sort == 'no-sort':
    for path in lines:
        if path.startswith('BR_no_sort_results'):
            pattern = "'(.*?)'"
            BR_results_directory = re.search(pattern, path).group(1)
        
    # Chosen system parameters
    S = 7
    BP = 50
    hist_mem = 2 # this variable doesn't matter for no sort architecture
    nb_enc = 1

elif approx_or_no_sort == 'approx-sort': 
    for path in lines:
        if path.startswith('BR_approx_sort_results'):
            pattern = "'(.*?)'"
            BR_results_directory = re.search(pattern, path).group(1) 
      
    # Chosen system parameters
    S = 3
    BP = 50
    hist_mem = 6 # bits
    nb_enc = 1
                

# We load all of the BR results, collate them, take a random sample of the channels' BRs
hist_mem_index = hist_mem-2 # since the hist sizes start at 2
x = np.zeros((nb_random_CVs,len(nb_channels_vec)))
raw_MUA_channels_power = np.zeros((len(nb_channels_vec)))
cv_count = 0
for CV in np.arange(1,5):
    cv_count += 1
    file_name = BR_results_directory + \
    '\BRs_S_' + str(S) + '_BP_' + str(BP)+'_CV_'+str(CV)+'.pkl'
    
    # Load BR results
    with open(file_name, 'rb') as file:
        results = pickle.load(file)
        stored_all_var_BRs = results.get('stored_all_var_BRs')
        encoder_red_rounds = len(stored_all_var_BRs) # reduction rounds
     
    # Get BR vector for all channels
    BRs = np.array(stored_all_var_BRs[encoder_red_rounds-nb_enc][hist_mem_index])
        
    # Take random sample of Z channels (for a range of Z values), get the 
    # summed BR, and from there the total implant power
    channel_vec = np.arange(len(BRs))
    for nb_channel_count, nb_channels in enumerate(nb_channels_vec):
        
        for random_CV in np.arange(nb_random_CVs):
            random_channel_subset = np.random.choice(channel_vec,nb_channels)
            BR_subset_sum = np.sum(BRs[random_channel_subset])
            
            temp = comm_energy * BR_subset_sum + nb_channels*(ADC_power + chan_processing_power) + static_process_power
            x[random_CV,nb_channel_count] += temp 
          
        raw_MUA_channels_power[nb_channel_count]  += nb_channels * (comm_energy * 1e3 + ADC_power + chan_processing_power) + static_process_power
    
# Take average across CV runs
raw_MUA_channels_power = raw_MUA_channels_power / cv_count
x = x / cv_count

# Find cases where the power budget was exceeded
nb_channels_that_exceeded_power_budget = x > total_power_budget

# Get permutation-derived p-value of how likely having Z channels is to exceed
# the power budget
xxx = np.vstack((nb_channels_vec,np.sum(nb_channels_that_exceeded_power_budget,axis=0)))
print('How many random channel combos (right) exceeded power budget for number Z of channels (size of each combo) (left): ',np.transpose(xxx))

comp_index = np.min(np.where(np.sum(nb_channels_that_exceeded_power_budget,axis=0)>0))-1
if comp_index == -1:
    print('Lower range of nb_channels_vec to get an estimate of the max nb of channels for the compressed data, since every considered number of channels exceeded the power budget for the compressed data')
else:
    nb_comp_MUA_channels = nb_channels_vec[comp_index]
    print('Comp MUA can achieve ',str(nb_comp_MUA_channels),' channels')

raw_index = np.min(np.where((raw_MUA_channels_power > total_power_budget)>0))-1
if raw_index == -1:
    print('Lower range of nb_channels_vec to get an estimate of the max nb of channels for the raw data, since every considered number of channels exceeded the power budget for the raw data')
else:
    nb_raw_MUA_channels = nb_channels_vec[raw_index]
    print('Raw MUA can achieve ',str(nb_raw_MUA_channels),' channels')
