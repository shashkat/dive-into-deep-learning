# One way of estimating cache size is to create numpy arrays of different sizes, and access
# elements from them randomly (so that caching is tested), and at whichever array size, the 
# average access time suddenly increases, is likely the cache size, and the whole array is 
# not being able to be cached.

import numpy as np
import time
import matplotlib.pyplot as plt

# function to access n items the array at different strides, and return the time taken to do so
def RecordTimeForAccess(stride = 1):
	# it is important to create a new array for each test, because we don't want that the array's
	# elements are already in cache, which might ruin our results.
	arr = np.ones(shape = 100000000, dtype = np.int8)

	starttime = time.perf_counter()
	# loop through certain values and we will access those indices in arr
	for i in range(0, 1+stride*10000, stride):
		_ = arr[i] # simple access of an entry in the array
	endtime = time.perf_counter()
	return endtime - starttime

# I try to access the elements of the array at different strides, specifically at strides of 
# 1 and 129, because I know for a fact that my cache line size is 128 bytes ($sysctl -a). Hence, 
# if continuous memory elements are loaded into memory for caching purpose, this should show 
# some difference in time taken
RecordTimeForAccess(1) 		# 0.0029707499779760838
RecordTimeForAccess(129) 	# 0.0027174160350114107
RecordTimeForAccess(1000) 	# 0.0026666249614208937

# Conclusion: Even though this experiment is well designed, the times taken seem to appear 
# similar because of the existence of CPU prefetching. Modern CPUs have hardware prefetchers 
# that try to predict the accessing pattern of data. In this case, the accessing pattern is 
# quite simple (increment of 129), and hence the prefetcher is able to load the appropriate
# data into l1 cache even though we try to fool it by having large strides. Ref - https://www.perplexity.ai/search/could-you-tell-me-about-the-di-F0DhUdefRD2YnVECpz3o7w#4
# A possible solution to this is to access the elements of the array randomly.

# So I will create a function in which we create an array just like in the prev function 
# RecordTimeForAccess, and then, we access the 0-indexed element from the array, and then 
# 31 random elements from the array. However, we would have the option to specify that these 
# 31 random indices are within what range of indices (and we make sure that there is no index 
# repeat, just to highlight a bit more, the effect of cached elements instead of same element 
# only). This way, we are fooling cpu prefetcher by accessing elements in random order, and 
# can highlighting cache effects by limiting the random indices to just the start of array, 
# and vice-versa.
# index_range_end - index_range_start should be greater than equal to 31
def RecordTimeForAccess2(index_range_start = 1, index_range_end = 128):
	arr = np.ones(shape = 1000000, dtype = np.int8)

	# create indices_to_access
	indices_to_access = [0] # start with appending the 0, so that upon the first access, the first 128 elements of the array are stored in l1 cache	
	# now, according to supplied parameter (range of values), choose random 31 elements
	all_possible_indices = np.arange(start = index_range_start, stop = index_range_end)
	# choose 31 random elements from all_possible_indices and append to indices_to_access
	indices_to_access += list(np.random.choice(a = all_possible_indices, size = 31, replace = False))

	starttime = time.perf_counter()
	# loop through certain values and we will access those indices in arr
	for i in indices_to_access:
		_ = arr[i] # simple access of an entry in the array
	endtime = time.perf_counter()
	return endtime - starttime

# now, lets record the time for 32 accesses in case of random indices but just from the 
# first 128 elements of the array
time_taken = 0
for _ in range(1000):
	time_taken += RecordTimeForAccess2()
time_taken  # 0.00712 for 1000 iterations
# now, when the 128 random elements can be from far away parts of the array
time_taken2 = 0
for _ in range(1000):
	time_taken2 += RecordTimeForAccess2(1, 100000) 
time_taken2 # 0.01410 for 1000 iterations

# now, lets try to vary the range from which we get 127 random elements in the array. Hopefully 
# this would increase gradually, starting at 128, and indicate to us that the cache line size 
# is 128 bytes
times_for_different_index_ends = []
# for index_range_end in [32, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 1000, 10000, 100000, ]:
for index_range_end in np.arange(start = 32, stop = 1000, step = 5):
	time_taken2 = 0
	for _ in range(1000):
		time_taken2 += RecordTimeForAccess2(1, index_range_end) 
	# print(f'{index_range_end}: {time_taken2}')
	times_for_different_index_ends.append(time_taken2)

plt.scatter(x = np.arange(start = 32, stop = 1000, step = 5), y = times_for_different_index_ends)
plt.show()

















### old stuff below


# 32: 0.0038936590535740834
# 48: 0.0038650729784421856
# 60: 0.0039011239914543694
# 72: 0.004056750965901301
# 84: 0.004123380002056365
# 96: 0.004136715007916791
# 108: 0.004131336003410979
# 120: 0.004223213998557185
# 132: 0.004408231996421819
# 144: 0.004408135035191663
# 156: 0.00449293400379247
# 168: 0.004643994972866494
# 180: 0.00464312201620487
# 192: 0.004762997035868466
# 204: 0.004779399976541754


# now, lets try to vary the range from which we get 127 random elements in the array. Hopefully 
# this would increase gradually, starting at 128, and indicate to us that the cache line size 
# is 128 bytes
times_for_different_index_ends = []
# for index_range_end in [32, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 1000, 10000, 100000, ]:
for index_range_end in np.arange(start = 32, stop = 2000, step = 5):
	time_taken2 = 0
	for _ in range(1000):
		time_taken2 += RecordTimeForAccess2(1, index_range_end) 
	# print(f'{index_range_end}: {time_taken2}')
	times_for_different_index_ends.append(time_taken2)

plt.clf()
plt.scatter(x = np.arange(start = 32, stop = 2000, step = 5), y = times_for_different_index_ends)
plt.show()



RecordTimeForAccess2(10)
RecordTimeForAccess(20)
RecordTimeForAccess(30)
RecordTimeForAccess(40)
RecordTimeForAccess(50)
RecordTimeForAccess(60)
RecordTimeForAccess(70)
RecordTimeForAccess(80)
RecordTimeForAccess(90)
RecordTimeForAccess(100)
RecordTimeForAccess(110)
RecordTimeForAccess(120)
RecordTimeForAccess(130)
RecordTimeForAccess(140)
RecordTimeForAccess(1000)

# accessing in series is not giving the results I would expect. Hence, think of some way of 
# estimating cache size using randomized access as then the biases are fewer.



# things are picking up after 1e6. Hence, we try to zoom in into that region
ys = []
n_values = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
for n in n_values:
	n = int(n)
	time_taken = AccessTimeOfArrayOfLengthN(n)
	print(f'{n}: {time_taken}')
	ys.append(time_taken)

plt.scatter(x = np.arange(len(ys)), y = ys)
plt.show()



# lets plot these results
ys = []
for n in [1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]:
	n = int(n)
	ys.append(AccessTimeOfArrayOfLengthN(n))


plt.scatter(x = np.arange(len(ys)), y = ys)
plt.show()

1e8/(4*1024*1024)

for n in [1e8]:
	n = int(n)
	print(f'{n}: {AccessTimeOfArrayOfLengthN(n)}')


# Function to take an input array length (n), and create a numpy array of that length, and 
# compute the average access time for accessing elements of that numpy array.
def AccessTimeOfArrayOfLengthN(n):
	arr = np.ones(shape = n, dtype = np.int8)

	# do 3 accesses of n elements from the array, and just consider the time for the third (the 
	# first 2 are warmup to remove biases from other processes doing one time intitiation for 
	# access from the array)
	
	times_for_n_accesses = []
	n_accesses_to_do = min(100, n)
	for _ in range (3):
		
		# get some random indices in which order we will access the elements of arr
		random_indices = np.random.randint(low = 0, high = n, size = n_accesses_to_do) # get n random indices

		starttime = time.perf_counter()
		# go through all the entries in random_indices array, and access that element of arr
		for i in random_indices:
			_ = arr[i]
		endtime = time.perf_counter()
		# append to times_for_n_accesses array, the time for this iteration out of the 3
		times_for_n_accesses.append(endtime - starttime)

	# return just the third time, divided by the number of accesses there were, so average
	return times_for_n_accesses[2]/n_accesses_to_do

for n in [1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]:
	n = int(n)
	print(f'{n}: {AccessTimeOfArrayOfLengthN(n)}')

# 1: 4.207948222756386e-06
# 10: 8.083996362984181e-07
# 100: 4.833401180803775e-07
# 1000: 3.917909925803542e-07
# 10000: 4.4133304618299007e-07
# 100000: 3.042499884031713e-07
# 1000000: 2.8175004990771413e-07
# 10000000: 4.924579989165067e-07
# 100000000: 2.5700000114738943e-06
# 1000000000: 0.0005695736670168117

# things are picking up after 1e6. Hence, we try to zoom in into that region
ys = []
n_values = np.arange(start = 1e2, stop = 10000*1e2+1, step = 5*1e2)
for n in n_values:
	n = int(n)
	time_taken = AccessTimeOfArrayOfLengthN(n)
	# print(f'{n}: {time_taken}')
	ys.append(time_taken)

plt.scatter(x = np.arange(len(ys)), y = ys)
plt.show()
plt.yscale('log')
plt.plot(ys)

# 1000000: 7.040000054985285e-07
# 2000000: 1.7791659920476377e-06
# 3000000: 1.871374959591776e-06
# 4000000: 8.27041978482157e-07
# 5000000: 2.0254170522093774e-06
# 6000000: 1.3202499831095339e-06
# 7000000: 1.4388749841600657e-06
# 8000000: 8.50957992952317e-07
# 9000000: 4.10795904463157e-06
# 10000000: 3.0264590168371796e-06
# 11000000: 7.859834004193545e-06
# 12000000: 8.867500000633299e-06
# 13000000: 3.814370901091024e-05
# 14000000: 1.144079101504758e-05
# 15000000: 1.1786249990109355e-05
# 16000000: 9.908791980706155e-06
# 17000000: 6.554582971148193e-06
# 18000000: 1.2524750025477261e-05
# 19000000: 7.866958039812744e-06
# 20000000: 1.1407250014599412e-05

# CONCLUSION: As we can see, it starts happening around 


import time
import numpy as np

def measure_access_time(array_size, iterations=1000):
    data = np.ones(array_size, dtype=np.int8)  # 1 byte per element
    start = time.perf_counter()
    for i in range(iterations):
        for j in range(array_size):
            _ = data[j]
    end = time.perf_counter()
    return (end - start) / (iterations * array_size)

def estimate_cache_size(max_size=100*1024, step=1024*10, threshold=1e-7):
    sizes = []
    times = []
    size = step
    while size <= max_size:
        t = measure_access_time(size)
        sizes.append(size)
        times.append(t)
        size += step
    # Detect jump in times to estimate cache boundary
    for i in range(1, len(times)):
        if times[i] > times[i-1] * (1 + threshold):
            print(f"Estimated cache boundary near {sizes[i-1]} bytes")
            break

estimate_cache_size()

# function to access n items the array at different strides, and return the time taken to do so
def RecordTimeForAccess(stride = 1, n_iters = 1000):
	arr = np.ones(shape = 1000000, dtype = np.int8)
	times = 0
	for _ in range(n_iters):
		starttime = time.perf_counter()
		# random index from where to start access
		startind = np.random.randint(1000000-10)
		# loop through certain values and we will access those indices in arr
		for i in range(startind, startind+stride, stride):
			_ = arr[i] # simple access of an entry in the array
		endtime = time.perf_counter()
		times += endtime - starttime
	return times/n_iters




