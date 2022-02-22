"""
Evaluating spike-based BMI decoding using Wiener filter
Load Flint data from matlab. HPC version.
"""

# import packages
import h5py
import numpy as np
from HPC_working_dir.functions.preprocess import input_shaping, split_index
from HPC_working_dir.functions.decoders import WienerCascadeDecoder
from HPC_working_dir.functions.metrics import compute_rmse, compute_pearson
import time as timer
import pickle

from scipy import io
import copy

import os
from os.path import exists


def moving_average(a, n=3) :
    a = np.hstack((np.zeros(n-1),a))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Create formatted data folder on HPC directory "working_directory", upload 
# formatted data (onlt the .mat files, all_binned_data.pkl not required) to it,
# and add the path below.
# Create a results_test_Sabes folder, and add the path below.
# Upload filenames_Sabes_test.txt to the HPC "working_directory".
    
def BDP_for_S_and_BP():
    
    # Path to working directory
    # working_directory = '/rds/general/user/ows18/home/MUA_CR_Sabes/'
    working_directory = '/rds/general/user/ows18/home/rerun_MUA_upload/'
    if working_directory == '':
        print('Fill in path to working directory')
        return 0
    
  
    file_names = working_directory + 'filenames_Sabes_test.txt'

    mat_folder = working_directory + 'neural_data/' # spike features folder
    result_folder = working_directory + 'results_test_Sabes/'              # results folder
        
    
    delta_time_vec = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    time_steps_vec = [5, 10, 15]
    lag_values_vec = [0, 5, 10]
    window_len_vec = [0, 0.05, 0.1, 0.2]
    alpha_vec = [0, 1e-4, 1e-2]
    degree_vec = [2,3,4]
    num_fold = 5 # number of folds
    regular = 'l2' # regularisation type
        
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
    print('PBS_ARRAY_INDEX: ' + str(PBS_ARRAY_INDEX))

    print ("Starting simulation")
    run_start = timer.time()
    
    feature_list = ['sua_rate', 'mua_rate']
    feature = feature_list[1]
        
    # Look at all stored file names, we have it this way so the indexing is 
    # consistent, always relative to the .txt file 
    with open(file_names) as f:
        lines = f.readlines()  
    
    # Rec indexing
    PBS_ARRAY_INDEX_rec = PBS_ARRAY_INDEX % len(lines)
    count_1 = int(np.floor(PBS_ARRAY_INDEX/(len(lines))))
    
    # BP indexing
    PBS_ARRAY_INDEX_BP = count_1 % len(delta_time_vec)
    count_2 = int(np.floor(count_1/(len(delta_time_vec))))
    
    # Wdw indexing
    PBS_ARRAY_INDEX_wdw = count_2 % len(window_len_vec)
    
    delta_time = delta_time_vec[PBS_ARRAY_INDEX_BP]
    file_name = lines[PBS_ARRAY_INDEX_rec].replace('\n','')
    wdw_time = window_len_vec[PBS_ARRAY_INDEX_wdw]
    
    print('BP: ' + str(delta_time*1000) + '; Rec-sub: ' + file_name + ' - wdw: ' + str(wdw_time))
        
    # Load neural data
    mat_filename = mat_folder+file_name + '_BP_'+ str(int(delta_time*1000))+'_ms.mat'  
    print ("Loading input features from file: "+mat_filename)
    f = io.loadmat(mat_filename)
    
    model = WienerCascadeDecoder() # instantiate model
              
    # Moving average window
    wdw_samples = int(np.round(wdw_time / delta_time))
    
    input_feature_1 = f['data'][:]
    cursor_vel_1 = f['cursor_pos_resamp'][:] # in mm/s
    
    print('input shape: ' + str(np.shape(input_feature_1)))
    print('output shape: ' + str(np.shape(cursor_vel_1)))

    for timesteps in time_steps_vec:
            
        input_feature = copy.deepcopy(input_feature_1)
        cursor_vel = copy.deepcopy(cursor_vel_1)
  
        input_dim = input_feature.shape[1] # input dimension
        output_dim = cursor_vel.shape[1] # output dimension NOTE CHANGE FOR ORIG

        # Initialise performance scores (RMSE and CC) with nan values
        rmse_valid = np.full((num_fold,output_dim),np.nan)
        rmse_test = np.copy(rmse_valid)
        cc_valid = np.copy(rmse_valid)
        cc_test = np.copy(rmse_valid)
        time_train = np.full((num_fold),np.nan)
        time_test = np.copy(time_train) 
        
        print ("Formatting input feature data")
        stride = 1 # number of samples to be skipped
        X_in = input_shaping(input_feature,timesteps,stride)
        X_in = X_in.reshape(X_in.shape[0],(X_in.shape[1]*X_in.shape[2]),order='F')
        
        print ("Formatting output (kinematic) data")
        diff_samp = cursor_vel.shape[0]-X_in.shape[0]
        Y_out = cursor_vel[diff_samp:,:] # in mm/s (remove it for new corrected velocity)

        print ("Splitting input dataset into training, validation, and testing subdataset")
        all_train_idx,all_valid_idx,all_test_idx = split_index(X_in,num_fold)
        
        S_vector = np.arange(2,40)
        copy_X_in = copy.deepcopy(X_in)
        for S in S_vector:
            X_in = copy.deepcopy(copy_X_in)
            
            # Clip dynamic range
            X_in[X_in>S] = S
            print('S: ' + str(S))
            
            # Moving average window
            if wdw_samples != 0:
                for channel in np.arange(len(X_in[0,:])):
                    X_in[:,channel] = moving_average(X_in[:,channel],wdw_samples)
                  

            for lag_value in lag_values_vec:
                lag = int(-0.004 / delta_time * lag_value) # lag between kinematic and ´neural data (minus indicates neural input occurs before kinematic)
              
                for alpha in alpha_vec:
                    for degree in degree_vec:
                        params = {'timesteps':timesteps, 'regular': regular, 'alpha':alpha, 'degree':degree}
                        
                        # Storing evaluation results into pkl file
                        result_filename = result_folder+file_name+\
                            '_delta_'+str(int(delta_time*1e3))+'ms_S_'+str(int(S))+\
                            '_wdw_' + str(int(wdw_time*1000)) + '_lag_'+str(lag_value)\
                            + '_timestep_'+str(timesteps) +\
                            '_alpha_' + str(alpha) + '_deg_' \
                            + str(degree) + '.pkl' # .h5
                            
                        if exists(result_filename):
                          print('Results exist \n')
                          continue
                        
                    
                        for i in range(num_fold):    
                            train_idx = all_train_idx[i]
                            valid_idx = all_valid_idx[i]
                            test_idx = all_test_idx[i]
                            
                            # specify training dataset
                            X_train = X_in[train_idx,:]            
                            Y_train = Y_out[train_idx,:]
                            
                            # specify validation dataset
                            X_valid = X_in[valid_idx,:]
                            Y_valid = Y_out[valid_idx,:]
                            
                            # specify validation dataset
                            X_test = X_in[test_idx,:]
                            Y_test = Y_out[test_idx,:]
                            
                            # Standardise (z-score) input dataset
                            X_train_mean = np.nanmean(X_train,axis=0)
                            X_train_std = np.nanstd(X_train,axis=0) 
                            X_train = (X_train - X_train_mean)/X_train_std 
                            X_valid = (X_valid - X_train_mean)/X_train_std 
                            X_test = (X_test - X_train_mean)/X_train_std 
                            
                            # Remove nan columns
                            remove = np.isnan(X_train[0,:])
                            X_train = np.delete(X_train,remove,1)
                            X_valid = np.delete(X_valid,remove,1)
                            X_test = np.delete(X_test,remove,1)
                            
                            # Zero mean (centering) output dataset
                            Y_train_mean = np.nanmean(Y_train,axis=0) 
                            Y_train = Y_train - Y_train_mean 
                            Y_valid = Y_valid - Y_train_mean
                            Y_test = Y_test - Y_train_mean
                            
                        
                            #Re-align data to take lag into account
                            if lag < 0:
                                X_train = X_train[:lag,:] # remove lag first from end (X lag behind Y)
                                Y_train = Y_train[-lag:,:] # reomve lag first from beginning
                                X_valid = X_valid[:lag,:]
                                Y_valid = Y_valid[-lag:,:]
                                X_test = X_test[:lag,:]
                                Y_test = Y_test[-lag:,:]
                            if lag > 0:
                                X_train = X_train[lag:,:] # reomve lag first from beginning
                                Y_train = Y_train[:-lag,:] # remove lag first from end (X lead in front of Y)
                                X_valid = X_valid[lag:,:]
                                Y_valid = Y_valid[:-lag,:]            
                                X_test = X_test[lag:,:]
                                Y_test = Y_test[:-lag,:]  
                                
                            print("Instantiating and training model...")    
                            model = WienerCascadeDecoder() # instantiate model
                            start = timer.time()
                            
                            model.fit(X_train,Y_train,**params) # train model
                            end = timer.time()
                            print("Model training took {:.2f} seconds".format(end - start))        
                            time_train[i] = end - start
                            print("Evaluating model...")
                            Y_valid_predict = model.predict(X_valid)
                            start = timer.time()
                            Y_test_predict = model.predict(X_test)
                            end = timer.time()
                            print("Model testing took {:.2f} seconds".format(end - start)) 
                            time_test[i] = end - start
                            
                            # Compute performance metrics    
                            rmse_vld = compute_rmse(Y_valid,Y_valid_predict)
                            rmse_tst = compute_rmse(Y_test,Y_test_predict)
                            cc_vld = compute_pearson(Y_valid,Y_valid_predict)
                            cc_tst = compute_pearson(Y_test,Y_test_predict)
                            rmse_valid[i,:] = rmse_vld
                            rmse_test[i,:] = rmse_tst
                            cc_valid[i,:] = cc_vld
                            cc_test[i,:] = cc_tst
                                
                            print("Fold-{} | Validation RMSE: {:.2f}".format(i,np.mean(rmse_vld)))
                            print("Fold-{} | Validation CC: {:.2f}".format(i,np.mean(cc_vld)))
                            print("Fold-{} | Testing RMSE: {:.2f}".format(i,np.mean(rmse_tst)))
                            print("Fold-{} | Testing CC: {:.2f}".format(i,np.mean(cc_tst)))
                        
                        run_end = timer.time()
                        mean_rmse_valid = np.nanmean(rmse_valid,axis=0)
                        mean_rmse_test = np.nanmean(rmse_test,axis=0)
                        mean_cc_valid = np.nanmean(cc_valid,axis=0)
                        mean_cc_test = np.nanmean(cc_test,axis=0)
                        mean_time =  np.nanmean(time_train,axis=0)
                        print("----------------------------------------------------------------------")
                        print("Validation Mean RMSE: %.3f " %(np.mean(mean_rmse_valid)))
                        print("Validation Mean CC: %.3f " %(np.mean(mean_cc_valid)))
                        print("Testing Mean RMSE: %.3f " %(np.mean(mean_rmse_test)))
                        print("Testing Mean CC: %.3f " %(np.mean(mean_cc_test)))
                        print("----------------------------------------------------------------------")
                        
                        # Store results
                        print ("Storing results into file: "+result_filename)     
                        with open(result_filename, 'wb') as file:
              
                          results = {'rmse_valid': rmse_valid,
                                     'rmse_test': rmse_test,
                                     'cc_valid': cc_valid,
                                     'cc_test': cc_test} # Shows how much of the validation data is used for assignment vs CR
                                     #'Y_true': Y_test, # Shows how much of the validation data is used for assignment vs CR
                                     #'Y_predict': Y_test_predict}
                          
                          # A new file will be created
                          pickle.dump(results, file)
                        
                        run_time = run_end - run_start
                        print ("Finished whole processes within %.2f seconds" % run_time)
    print("All done")
