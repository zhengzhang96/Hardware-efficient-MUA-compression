# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 03:53:30 2022

Basically, we're checking how mnay channels we cna have in FPGA. Some channels
have higher BRs that others. So what's the worst case? Take random sample of
all considered channels, take the total BR, and add the ADC and static power,
comm enerry per bit, ge tthe total power. If it exceeds the heating limit, 
mark it (matrix called temp at the end). Each column in temp corresponds to a 
number of considered channels, given in 'nb_channels_vec'. This basically gives
the p-value.

Could try low-power architecture too.

@author: ows18
"""
import pickle
import numpy as np
#from random import sample

## Integrate our results with Zheng's
CR_res_folder = 'D:\\Dropbox (Imperial NGNI)\\NGNI Share\\Workspace\\'+\
    'Oscar\\Work\\MUA compression\\End days'
   
with_sort = True
# With sort
if with_sort == True:
    file_name_data = CR_res_folder + '\\CR results zheng sort\\End_days_2_zheng_sort_CRs_S_'
    S = 3
    BP = 50
    hist_mem = 6 # bits

else:
    # Without sort
    file_name_data = CR_res_folder + '\\CR results no sort\\End_days_2_CRs_S_'
    S = 7
    BP = 50
    hist_mem = 2 # doesn't matter for no sort


nb_channels_vec = np.arange(290, 320)
#nb_channels_vec = [27,28,29,30,31]

nb_random_CVs = 10000

static_process_power = 0.1618e-3 
chan_processing_power = 0.96e-6
comm_energy = 20e-9
ADC_power = 0 # 1e-6 # count it as off-FPGA
#total_power_budget = 10e-3 * (1.4e-1 * 1.48e-1)
total_power_budget = 10e-3 * (2.5e-1 * 2.5e-1)

hist_mem_index = hist_mem-2 # since the hist sizes start at 2



x = np.zeros((nb_random_CVs,len(nb_channels_vec)))
raw_MUA_channels_power = np.zeros((len(nb_channels_vec)))

cv_count = 0
for CV in np.arange(1,29):
    cv_count += 1
#CV = 1
    file_name = file_name_data + \
    str(S) + '_BP_' + str(BP)+'_CV_'+str(CV)+'.pkl'
    
    
    with open(file_name, 'rb') as file:
    # Call load method to deserialze
        results = pickle.load(file)
        stored_all_var_CRs = results.get('stored_all_var_CRs')
        encoder_red_rounds = len(stored_all_var_CRs) # reduction rounds
        
    
    BRs = 1000/np.array(stored_all_var_CRs[-1][hist_mem_index])[:,-1] # take last CV reduction roun, and CR relative to 1ms estimated
        
    channel_vec = np.arange(len(BRs))
    
    for nb_channel_count, nb_channels in enumerate(nb_channels_vec):
        
        for random_CV in np.arange(nb_random_CVs):
            random_channel_subset = np.random.choice(channel_vec,nb_channels)
            BR_subset_sum = np.sum(BRs[random_channel_subset])
            
            temp = comm_energy * BR_subset_sum + nb_channels*(ADC_power + chan_processing_power) + static_process_power
            x[random_CV,nb_channel_count] += temp 
          
        raw_MUA_channels_power[nb_channel_count]  += nb_channels * (comm_energy * 1e3 + ADC_power + chan_processing_power) + static_process_power
        #input('yeet')
    
raw_MUA_channels_power = raw_MUA_channels_power / cv_count
x = x / cv_count
temp = x > total_power_budget

nb_comp_MUA_channels = nb_channels_vec[np.min(np.where(np.sum(temp,axis=0)>0))-1]
print('How many random combos exceeded power budget: ',np.sum(temp,axis=0))
print('Comp MUA can achieve ',str(nb_comp_MUA_channels),' channels')
nb_raw_MUA_channels = nb_channels_vec[np.min(np.where((raw_MUA_channels_power > total_power_budget)>0))-1]
print('Raw MUA can achieve ',str(nb_raw_MUA_channels),' channels')

# =============================================================================
# input('yeet gjsdjs')
# 
# # How few channels can we get
# BRs = 1000/np.array(stored_all_var_CRs[-1][0])[:,-1]
# channel_vec = np.arange(len(BRs))
# x = np.zeros((nb_random_CVs,1))
# nb_channels = 1
# nb_random_CVs = len(BRs)
# for random_CV in np.arange(nb_random_CVs):
#     random_channel_subset = np.random.choice(channel_vec,nb_channels)
#     BR_subset_mean = np.sum(BRs[random_channel_subset])
#     
#     temp = comm_energy * BR_subset_mean + nb_channels*(ADC_power+chan_processing_power) + static_process_power
#     x[random_CV] = temp 
#     
# print('Min square length of an implant for 1 channel: ',np.sqrt(max(x)/10e-3)[0],' cm')
# =============================================================================

      
