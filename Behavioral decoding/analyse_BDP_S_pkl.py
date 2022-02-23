# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:32:01 2021
Basically we need to find the optimal parameters using the validation results,
and index the test results for the same parameters.

@author: oscar
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re



# To plot in seperate window in Spyder
# %matplotlib qt


# Specify root directory (where directories.txt file is located)
root_directory = r'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload code'

##########################################################################


# Read directories.txt file
with open(root_directory + '\\directories.txt') as f:
    lines = f.readlines()

# Get path to BDP results
for path in lines:
    if path.startswith('BDP_results'):
        pattern = "'(.*?)'"
        BDP_results_directory = re.search(pattern, path).group(1)
     
# Get SCLV directory
for path in lines:
    if path.startswith('SCLV_path'):
        pattern = "'(.*?)'"
        SCLV_directory = re.search(pattern, path).group(1)
        

BP_vec = [1,5,10,20,50,100]
S_vector = np.arange(2,40)
time_steps_vec = [5, 10, 15]
lag_values_vec = [0, 5, 10]
window_len_vec = [0, 0.05, 0.1, 0.2]
alpha_vec = [0, 1e-4, 1e-2]
degree_vec = [2, 3, 4]


for Sabes_or_Flint in ['Flint']: #'Sabes',
    for train_or_test in ['test']: #'train',
        
        print(Sabes_or_Flint + ' - ' + train_or_test)

        if Sabes_or_Flint == 'Sabes':
            # Sabes, test
            if train_or_test == 'test':
                file_names = root_directory + '\\filenames_Sabes_test.txt'
                result_folder = BDP_results_directory + '\\results_test_Sabes\\'
                
            # Sabes, train
            if train_or_test == 'train':
                file_names = root_directory + '\\filenames_Sabes_train.txt'
                result_folder = BDP_results_directory + '\\results_train_Sabes\\'
    
        elif Sabes_or_Flint == 'Flint': 
            # Flint, test
            if train_or_test == 'test':
                file_names = root_directory + '\\filenames_Flint_test.txt'
                result_folder = BDP_results_directory + '\\results_test_Flint\\'
            
            # Flint, train
            if train_or_test == 'train':
                file_names = root_directory + '\\filenames_Flint_train.txt'
                result_folder = BDP_results_directory + '\\results_train_Flint\\'
        
        # Look at all stored file names, we have it this way so the indexing is
        # consistent, always relative to the .txt file
        with open(file_names) as f:
            lines = f.readlines()
        
        # Validation results
        all_results = np.zeros((len(lines), len(BP_vec), len(time_steps_vec),
                               len(lag_values_vec), len(
                                   window_len_vec), len(alpha_vec),
                                len(degree_vec), 60))
        
        # Test results
        all_test_results = np.zeros((len(lines), len(BP_vec), len(time_steps_vec),
                                     len(lag_values_vec), len(
                                         window_len_vec), len(alpha_vec),
                                     len(degree_vec), 60))
        
        for filename_index in np.arange(len(lines)):
            
            file_name = lines[filename_index].replace('\n','')
        
            for delta_count, delta_time in enumerate(BP_vec):
                for time_step_counter, timesteps in enumerate(time_steps_vec):
                    for lag_c, lag_value in enumerate(lag_values_vec):
                        for window_c, wdw in enumerate(window_len_vec):
                            for a_count, alpha in enumerate(alpha_vec):
                                for degree_count, degree in enumerate(degree_vec):
                                    for S in S_vector:
        
                                        try: # If result exists for that param combo, some jobs in HPC failed
                                        
                                            # Storing evaluation results into hdf5 file
                                            result_filename = result_folder+file_name +\
                                                '_delta_'+str(int(delta_time))+'ms_S_'+str(int(S)) +\
                                                '_wdw_' + str(int(wdw*1000)) + '_lag_'+str(lag_value)\
                                                + '_timestep_'+str(timesteps) +\
                                                '_alpha_' + str(alpha) + '_deg_' \
                                                + str(degree) + '.pkl'
         
                                            with open(result_filename, 'rb') as file:      
                                                results = pickle.load(file)
                                                result = np.mean(results['cc_valid'])
                                                test_result = np.mean(results['cc_test'])
                                                
                                            
                                            # Validation results
                                            # NOTE: we assume no information is
                                            # lost from increasing S, so BDP 
                                            # should never decrease from
                                            # increasing S, so we take the max.
                                            all_results[filename_index, delta_count,
                                                        time_step_counter, lag_c, window_c, a_count,
                                                        degree_count, S] = np.max(np.hstack((result,all_results[filename_index, delta_count,
                                                                    time_step_counter, lag_c, window_c, a_count,
                                                                    degree_count, :])))
                
                                            # Test results
                                            all_test_results[filename_index, delta_count,
                                                             time_step_counter, lag_c, window_c, a_count,
                                                             degree_count, S] = np.max(np.hstack((test_result,all_test_results[filename_index, delta_count,
                                                                              time_step_counter, lag_c, window_c, a_count,
                                                                              degree_count, :])))

                                        except: # If the S/BP combo doesn't exist, replace with max value from a smaller S value for the same BP value
                                            # Validation results
                                            all_results[filename_index, delta_count,
                                                        time_step_counter, lag_c, window_c, a_count,
                                                        degree_count, S] = np.max(all_results[filename_index, delta_count,
                                                                    time_step_counter, lag_c, window_c, a_count,
                                                                    degree_count, :])
        
                                            # Test results
                                            all_test_results[filename_index, delta_count,
                                                             time_step_counter, lag_c, window_c, a_count,
                                                             degree_count, S] = np.max(all_test_results[filename_index, delta_count,
                                                                              time_step_counter, lag_c, window_c, a_count,
                                                                              degree_count, :])
        
        
        print('Formatting done \n')
        
        # Finding the best parameters for each BP/S combo
        best_res = np.zeros((len(BP_vec), np.max(S_vector)+1))
        best_params = np.zeros((len(BP_vec), np.max(S_vector)+1, len(lines), 5))
        for BP_count, delta_time in enumerate(BP_vec):
            for S in S_vector:
                all_temp = 0
                all_temp_counter = 0
                for rec in np.arange(len(lines)):
                    xx = all_results[rec, BP_count, :, :, :, :, :, S]
        
                    # Find index of maximum value from 2D numpy array
                    y = np.where(xx == np.nanmax(xx))
                    
                    temp = np.copy(xx)
                    for yc, yy in enumerate(y): # iterating through axes
        
                        try:
                            if np.nanmax(xx) != 0:  # If max is not 0, should be a unique solution
                                best_params[BP_count, S, rec, yc] = yy[0]
                            temp = temp[yy[0]]  # reduce to smaller set
                        except: # Occurs if yy is empty becaus eonly NaNs in the slaice
                            best_params[BP_count, S, rec, yc] = float('NaN')
                            temp = 0
                    
                    # If we have an actual BDP value
                    if temp != 0: 
                        all_temp_counter += 1
                        all_temp += temp
                    
                if all_temp != 0: # keep from dividing 0 by 0
                    all_temp = all_temp / all_temp_counter
                    best_res[BP_count, S] += all_temp
                else:
                    best_res[BP_count, S] += best_res[BP_count, S-1]
        
        
        # Use validated params to get test results
        best_val_val_params = np.zeros((len(BP_vec), np.max(S_vector)+1, len(lines)))
        best_test_val_params = np.zeros((len(BP_vec), np.max(S_vector)+1, len(lines)))
        for delta_count, delta_time in enumerate(BP_vec):
            for S in S_vector:
                xx = 0
                xx_count = 0
                xx_test =  0
                xx_test_count = 0
                for rec in np.arange(len(lines)):
                    params = best_params[delta_count, S, rec]
        
                    # Verify results match, make sure param indexing is correct
                    # Validation results
                    try:
                        temp_val = all_results[rec, delta_count, int(params[0]), int(params[1]),
                                          int(params[2]), int(params[3]), int(params[4]), S]
                        
                    except: # If params don't exist, if it's NaN (no file for that combo)
                        temp_val = float('NaN')
                        
                    if not np.isnan(temp_val) and temp_val != 0:
                        xx_count += 1
                        xx += temp_val
                        
                    # Test results, same params
                    try:
                        temp_test = all_test_results[rec, delta_count, int(params[0]), int(params[1]),
                                          int(params[2]), int(params[3]), int(params[4]), S]
                    except:
                        temp_test = float('NaN')
                        
                
                    # Get validation performance using optimal parameters for each rec
                    best_val_val_params[delta_count, S, rec] = temp_val
                    
                    # Test results with the same
                    best_test_val_params[delta_count, S, rec] = temp_test
                        

        # Make sure each S is at least as good as all S before it
        for BP_counter, BP in enumerate(BP_vec):
            for rec in np.arange(len(lines)):
                for S in S_vector:
                    best_val_val_params[BP_counter,S,rec] = np.max(best_val_val_params[BP_counter,1:S+1,rec] )
                    best_test_val_params[BP_counter,S,rec] = np.max(best_test_val_params[BP_counter,1:S+1,rec] )
        

        # Store (hyper-)paramater optimised results, so we can load them 
        # easily for later use
        with open(BDP_results_directory+'\\S_vs_BDP_'+train_or_test+'_'+Sabes_or_Flint + '.pkl', 'wb') as file:
            results = {'best_test_val_params':best_test_val_params,
                       'best_val_val_params': best_val_val_params}
              
            # A new file will be created
            pickle.dump(results, file)
        
        
        
        # Boxplot for each BP, show boxplot of each S with label of nb of 
        # succesful recordings
        S_vector_plot = np.arange(40)
        
        for BP_counter, BP in enumerate(BP_vec):
            fig = plt.figure()
            b = np.transpose(best_test_val_params[BP_counter,:,:])
            
            plt.boxplot(b, positions = S_vector_plot)
        
            #plt.plot(np.transpose(best_test_val_params)) #best_test_val_params
            plt.xlabel('S')
            plt.ylabel('BDP')
            plt.legend(BP_vec)
            plt.title(Sabes_or_Flint + ' ' + train_or_test + ' ; BP:' + str(int(BP)) + ' ms')
            plt.show()
        

