# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:32:01 2021

@author: oscar
Script for plotting 3d scatter plots of results, so one can get an idea of
the interactions between BP, BDP, and FPGA power consumtion and resources.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import re
from openpyxl import load_workbook
import string

# Library for converting excel formulas into Python
from xlcalculator import ModelCompiler, Model, Evaluator


#################### SET ROOT DIRECTORY ##################################
# Specify root directory (where directories.txt file is located)
root_directory = r'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload code'
# %matplotlib qt
##########################################################################

# Read directories.txt file
with open(root_directory + '\\directories.txt') as f:
    lines = f.readlines()

# Get excel spreadsheet directory (has the hardware processing power and 
# resources results, in this script we we will add the BR and BDP results)
for path in lines:
    if path.startswith('combined_results_excel_path'):
        pattern = "'(.*?)'"
        excel_directory = re.search(pattern, path).group(1)
        excel_path = excel_directory + '\combined_results_template.xlsx'
     
# Load excel spreadsheet template, from it we get all of the results
workbook = load_workbook(filename=excel_path)
sheet = workbook.active

compiler = ModelCompiler()
new_model = compiler.read_and_parse_archive(excel_path)
evaluator = Evaluator(new_model)

# First and last rows and columns in excel spreadsheet
first_row = 3
last_row = 275
min_col = 1
max_col = 21

excel_results = np.zeros((last_row-first_row,max_col-min_col))
for column in np.arange(max_col-min_col):
    col = string.ascii_uppercase[column]
    
    for row_index, row in enumerate(np.arange(first_row,last_row)):
    
        val1 = evaluator.evaluate("Sheet1!"+col+str(row))
        value = str(val1)
        if value != '':
            excel_results[row_index,column] = float(value)
        else:
            excel_results[row_index,column] = float('NaN')
   
# Remove NaN values (rows for which some key value, e.g. BDP, is missing, so 
# no point plotting it)
for row in np.arange(len(excel_results[:,0])-1,-1,-1):
    if np.isnan(np.sum(excel_results[row,:])):
        excel_results = np.delete(excel_results,row,axis=0)         

# Iterate through different system configurations, and collate them so we can plot
# them all together
configs = ['no sort-map','full','just-bin']
for config_ind, config in enumerate(configs):
            
    # Architecture label
    temp_architecture = []
    for kk in np.arange(len(excel_results[:,0])):
        temp_architecture.append(config)
    temp_architecture = np.asarray(temp_architecture)
    temp_architecture = np.reshape(temp_architecture,(len(temp_architecture),1))
    
    # System config + parameters (S, hist size, #enc)
    temp_architecture = np.hstack((temp_architecture,excel_results[:,1:4]))
   

    if config_ind == 0:
        BP_coords = excel_results[:,0]
        BDP_coords = excel_results[:,4]
        processing_power = 0.96 # processing power
        
        # System config + parameters (S, hist size, #enc)
        architecture = copy.deepcopy(temp_architecture)
         
    else:
        BP_coords = np.hstack((BP_coords,excel_results[:,0]))
        BDP_coords = np.hstack((BDP_coords,excel_results[:,4]))
        architecture = np.vstack((architecture,temp_architecture))  # System config + parameters (S, hist size, #enc)
        

    if config == 'no sort-map':
        resource_coords = excel_results[:,10]
        BR_coords = excel_results[:,13]
        comm_power_coords = excel_results[:,18]
        
        
    elif config == 'full':
        resource_coords = np.hstack((resource_coords,excel_results[:,11]))
        BR_coords = np.hstack((BR_coords,excel_results[:,12]))
        comm_power_coords = np.hstack((comm_power_coords,excel_results[:,17]))
        
        
    elif config == 'just-bin':
        resource_coords = np.hstack((resource_coords,excel_results[:,5]))
        BR_coords = np.hstack((BR_coords,excel_results[:,14]))
        comm_power_coords = np.hstack((comm_power_coords,excel_results[:,19]))
        
   
power_coords = comm_power_coords + processing_power   
excel_results_stacked = np.transpose(np.vstack((BP_coords,resource_coords,power_coords,BDP_coords,BR_coords)))


################### 3d scatter plot ###########################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
p = ax.scatter(resource_coords, power_coords, BDP_coords, c =BP_coords, alpha = 0.8)
ax.set_xlabel('Resources')
ax.set_ylabel('Total power per channel (uW)')
ax.set_zlabel('BDP')
fig.colorbar(p, label="BDTP (ms)",ax = [ax], location='left')
plt.title('(a)')
plt.show()

# Find only table entries with 50 ms BP and less than 2.2 uW per channel and 
# 250 FPGA resources
low_power_coords = np.where(power_coords<2.2)[0]
low_resources_coords = np.where(resource_coords<250)[0]
BP_50_coords = np.where(BP_coords==50)[0]
desired_coords = np.intersect1d(low_power_coords,low_resources_coords)
desired_coords = np.intersect1d(desired_coords,BP_50_coords)
all_desired_res_stacked = np.hstack((architecture,excel_results_stacked))
narrowed_down_res = all_desired_res_stacked[desired_coords]

# Lowest power
chosen_index = desired_coords[np.argmin(power_coords[desired_coords])]

# Lowest resources
# chosen_index = desired_coords[np.argmin(resource_coords[desired_coords])]

plotted_desired_coords= np.setdiff1d(desired_coords,chosen_index)
plotted_desired_coords = np.insert(plotted_desired_coords,-1,chosen_index)
print('Lowest power:')
print('Architecture (with S, hist size, #enc): ',architecture[chosen_index])
print('BP, Resources, Power, BDP, BR = ',excel_results_stacked[chosen_index,:])



################### 2nd scatter plot ###########################
# Plot of a narrower (optimal) section of results, all at 50 ms BP
# Plot desired coords only (in chosen range)
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


fig, ax = plt.subplots()
m = np.repeat(["o"], len(plotted_desired_coords)-1)
edgecolors = np.repeat(["none"], len(plotted_desired_coords)-1)
size = np.repeat(40, len(plotted_desired_coords)-1)
m = np.insert(m,-1,'^')
edgecolors = np.insert(edgecolors,-1,'k')
size = np.insert(size,-1,60)
scatter = mscatter(resource_coords[plotted_desired_coords], power_coords[plotted_desired_coords], c =BDP_coords[plotted_desired_coords], s = size, m=m,edgecolors= edgecolors, ax=ax)
ax.set_xlabel('Resources')
ax.set_ylabel('Total power per channel (uW)')
fig.colorbar(scatter, label="BDP",ax = [ax])
plt.title('(b)')
plt.grid()
plt.show()
