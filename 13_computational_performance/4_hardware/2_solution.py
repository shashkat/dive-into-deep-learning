# In order to test the difference in accessing memory in a sequential manner vs in a strided 
# manner, I create a huge numpy array, and then time accessing a number of elements from it 
# with different strides.

import numpy as np
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# I will keep the dtype of the array as int. I will try to create an array of size in memory as 
# 1000mb, as it exceeds most cache sizes, so we can see nicely, the difference when caching 
# doesn't help. Length of array = 1000*1024*1024/4 = 26214400. 
arr = np.arange(start=0, stop = 262144000, dtype = int)

# function to access n items the array at different strides, and return the time taken to do so
def RecordTimeForAccess(arr, stride = 1, n_accesses = 1000):
	temp = 0
	access_number = 0
	starttime = time.perf_counter()
	# loop through certain values and we will access those indices in arr
	for i in range(0, len(arr), stride):
		temp = arr[i] # simple access of an entry in the array
		access_number += 1
		# if we have accessed the specified number of values from arr, then come out of loop
		if access_number == n_accesses:
			print('finished required number of accesses')
			break
	endtime = time.perf_counter()
	return round(endtime-starttime, 5)

RecordTimeForAccess(arr, 1) # 0.0007
RecordTimeForAccess(arr, 1000) # 0.001
RecordTimeForAccess(arr, 10000) # 0.001
RecordTimeForAccess(arr, 100000) # 0.001

# In the first calls, the difference in time taken is visible, but with more calls it disappears. 
# This is likely because the data we are accessing again and again is getting loaded in cache, 
# and hence next calls to it are faster.

# function to access n items the array either randomly or in sequence with stride of 1, and 
# return the time taken to do so
def RecordTimeForAccess2(arr, randomized = False, n_accesses = 10000):
	temp = 0
	access_number = 0
	random_indices = np.random.randint(low = 0, high = len(arr), size = n_accesses) # get n_accesses random indices
	if randomized == False:
		starttime = time.perf_counter()
		# loop through certain values and we will access those indices in arr
		for i in range(n_accesses):
			temp = arr[i] # simple access of an entry in the array
			access_number += 1
			# if we have accessed the specified number of values from arr, then come out of loop
			if access_number == n_accesses:
				print('finished required number of accesses')
				break
		endtime = time.perf_counter()
	else:
		starttime = time.perf_counter()
		# loop through certain values and we will access those indices in arr
		for i in random_indices:
			temp = arr[i] # simple access of an entry in the array
			access_number += 1
			# if we have accessed the specified number of values from arr, then come out of loop
			if access_number == n_accesses:
				print('finished required number of accesses')
				break
		endtime = time.perf_counter()
	return round(endtime-starttime, 4)

RecordTimeForAccess2(arr, False) # ~ 0.004 seconds
RecordTimeForAccess2(arr, True) # ~ 1-2 seconds

# Hence, when we access some random indices from the array, caching is not able to make it fast, 
# and the time taken goes up significantly, showing how accessing memory in sequence is much 
# faster, than accessing it in strides or in the worst case, randomly. 

