# to test this, I can try to set the number of threads to 2 maybe, and then see the time taken 
# for matrix mutiplications of different sizes to compute compared to when I have just 1 thread.

import torch
import time

# function to record time taken (with first 3 iterations not recorded for warmup)
def TenMatrixMultiplicationTime(x):
	times = []
	for step_idx in range(13):
		if step_idx < 3: continue
		starttime = time.time()
		torch.mm(x, x)
		endtime = time.time()
		times.append(endtime - starttime)
	return (round(sum(times), 5))

### WITH THREADS = 1
# first set the num_threads if not already to 1
torch.set_num_threads(1)
x = torch.randn(size=(10,10))
TenMatrixMultiplicationTime(x) # 0.00039
x = torch.randn(size=(100,100))
TenMatrixMultiplicationTime(x) # 0.05584
x = torch.randn(size=(1000,1000))
TenMatrixMultiplicationTime(x) # 0.335

### WITH THREADS = 2
torch.set_num_threads(2)
x = torch.randn(size=(10,10))
TenMatrixMultiplicationTime(x) # 0.00099
x = torch.randn(size=(100,100))
TenMatrixMultiplicationTime(x) # 0.01883
x = torch.randn(size=(1000,1000))
TenMatrixMultiplicationTime(x) # 0.09732

# CONCLUSION: Hence, we can see that for extrement small operations, it is better to do them using
# just 1 thread. However, for bigger matrix multiplications, using multiple threads can be quite
# beneficial
# Also note: The >2× speedup from 1 to 2 threads occurs because matrix multiplication performance 
# is not purely linear in thread count. Adding a second thread often improves cache efficiency, 
# memory latency hiding, and instruction-level parallelism, producing superlinear scaling for 
# certain matrix sizes and CPU architectures












