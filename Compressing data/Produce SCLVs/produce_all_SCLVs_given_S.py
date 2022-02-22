# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:35:45 2021

@author: oscar
"""
# Producing all SCLVs for each S.

import time
import numpy as np
import heapq
import time
from collections import defaultdict
import pickle


# Huffman encoder function, trained based on a probability distribution "frequency"
def encode(frequency):
    heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for value in low[1:]:
            value[1] = '0' + value[1]
        for value in high[1:]:
            value[1] = '1' +value[1]
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


# Script to produce all non-redundant SCLVs for each considered S value, given small 
# enough increment (related to the resolution of the probability distribution
# used to train the Huffman encoders, which are then reduced to SCLV 
# representation).

#folder = 'D:\\Dropbox (Imperial NGNI)\\NGNI Share\\Workspace\\Oscar\\Work\\MUA compression\\MUA_compression_1_Python_backup\\'

increment = 0.15 # the smaller the increment, the more likely we are to get every possible SCLV. However, it takes longer to run the algorithm.
all_CLVs = []
        
# Iterate through S values
for S in np.arange(2,11,1):
    
    p = np.zeros((S,1)) # prob. vector of length S
    #s = np.arange(S) # 
    frequency = defaultdict(int)
    CLV = []
    stored_LUTs = 0
    
    # To time how long the algorithm takes
    tic = time.perf_counter()
    
    counter = 0
    while p[-1][0] < 1: # until we've gone through every non-redundant combination
    
        # Update prob vector, iterate through possibilities
        # Basic idea is that if there are multiple values equal to the minimum,
        # the most significant one gets incremented, and all the rest are set to 0.
        # This makes sure we get all of the non-redundant possibilities
        # (with the exception of all values == the same, which happens as many times
        # as there are increments, but it's a redundancy that doesn't matter)
        min_val = np.min(p)
        a = np.where(p==min_val)
        for LHS in a[0][1:]:
            p[LHS] = 0
        p[a[0][0]] += increment
        counter += 1
    
        # Normalize probabilities to sum to 1
        p2 = p/np.sum(p)
        for k, prob in enumerate(p2):
            frequency[k] = prob
            
        # Train Huffman encoder (referred to as LUT)
        LUT = encode(frequency)
        temp = np.array(LUT)[:,1] # codeword lengths
  
        # Store encoders
        if stored_LUTs == 0: # first round, it's empty
            stored_LUTs = temp
        else:
            stored_LUTs = np.vstack((stored_LUTs,temp))
            
        # Reduce down to SCLV representation. CLVs are already sorted because
        # of the way the probability distribution was produced.
        CLV_temp = np.zeros((1,S))
        for k, clv in enumerate(temp):
            CLV_temp[0,k] = len(clv)
        
        # Add to SCLV list if we haven't seen the same vector before
        seen_b4 = False
        for x in CLV:
            if np.sum(np.abs(x-CLV_temp)) == 0:
                seen_b4 = True
                break
        if seen_b4 == False:
            CLV.append(CLV_temp[0])
    
    toc = time.perf_counter()
    print(f"Computing embeddings took {toc - tic:0.4f} seconds")
    

    # Open a file and use dump()
    with open('Stored_SCLVs_S_'+str(int(S))+'.pkl', 'wb') as file:
          
        # A new file will be created
        pickle.dump(CLV, file)
        
    all_CLVs.append(CLV)

new_array = [tuple(row) for row in stored_LUTs]
uniques = np.unique(new_array,axis=0)