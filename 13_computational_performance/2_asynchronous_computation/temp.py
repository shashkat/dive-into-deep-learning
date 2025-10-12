# just trying out some of python's default asynchronous computation and that offered by pytorch.

# first, we check if python by default has some kind of behaviour as is suggested in the chapter.
# i.e. having a frontend which just sends commands to the backend, and the backend which executes
# the commands, and python returning once the backend's results are computed (which are possibly 
# parallelized)

import sys
from d2l import torch as d2l
import torch
import time

x=0
%time for _ in range(1000000): y = x + 1; print(y)
# CPU times: user 1.09 s, sys: 442 ms, total: 1.53 s
# Wall time: 1.97 s

x=0
%time for _ in range(1000000): y = x + 1; print('hello')
# CPU times: user 1.06 s, sys: 450 ms, total: 1.51 s
# Wall time: 1.96 s

with d2l.Benchmark():
	for _ in range(10):
		a = torch.randn(size = (100,100))
		b = torch.mm(a, a)
	torch.cpu


starttime = time.time()
for _ in range(1000):
	a = torch.randn(size = (100,100), device = torch.device('mps'))
	b = torch.mm(a, a)
torch.mps.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.171

print(torch.get_num_threads())
torch.set_num_threads(8)

starttime = time.time()
for _ in range(10):
	a = torch.randn(size = (1000,1000), device = torch.device('cpu'))
	b = torch.mm(a, a)
torch.cpu.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.220

torch.set_num_threads(8)
starttime = time.time()
for _ in range(10):
	a = torch.randn(size = (1000,1000), device = torch.device('cpu'))
	b = torch.mm(a, a)
torch.cpu.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.269

torch.set_num_threads(1)
starttime = time.time()
for _ in range(10):
	a = torch.randn(size = (1000,1000), device = torch.device('cpu'))
	b = torch.mm(a, a)
torch.cpu.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.581

## SO BY SETTING THE NUMBER OF THREADS TO 8 (WHICH WAS DEFAULT FOR SOME REASON), PYTORCH 
# AUTOMATICALLY TRIES TO PARALLELIZE THE TASK. THIS I INFERRED by LOOKING AT THE TIME TAKEN 
# FOR THE TASKS. NEXT GETTING TO KNOW A BIT MORE INTO HOW THE PARALLELIZATION IS DONE, HOW TO 
# BREAK IT ETC. AND OVERALL COMPARING WITH THE PARALLELIZATION WE DID IN P4S

# 1) first, seeing if printing at end of each iteration will make it slow (as then the frontend 
# would need to wait at end of each iteration for the result and only then be able to move to 
# the next iteration)

# base case with printing hello
torch.set_num_threads(8)
starttime = time.time()
for _ in range(10):
	a = torch.randn(size = (1000,1000), device = torch.device('cpu'))
	b = torch.mm(a, a)
	print('hello')
torch.cpu.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.286

# actual case with printing value of b
torch.set_num_threads(8)
starttime = time.time()
for _ in range(10):
	a = torch.randn(size = (1000,1000), device = torch.device('cpu'))
	b = torch.mm(a, a)
	print(b)
	# torch.cpu.synchronize()
torch.cpu.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.285

## OKAY, SO PRINTING B AT THE END OF EACH ITERATION DIDN'T MAKE THE CODE SLOWER. THIS IS BECAUSE 
# USAGE OF MULTIPLE THREADS IS HAPPENING IN A DIFFERENT WAY HERE THAN I WAS EXPECTING. ACTUALLY 
# PYTORCH IS JUST USING THE MANY THREADS FOR INTRA-OPERATION USAGE (SO THE MATRIX MULTIPLICATION 
# OPERATION ONLY IS USING MULTIPLE THREADS IF AVAILABLE, BUT EVERYTHING ELSE HERE IS BY DEFAULT 
# BLOCKING ON CPU (MEANING NO PARALLELIZATION)). HENCE WE SEE NO EFFECT IN RUNTIME EVEN IF WE 
# PRINT B AT THE END OF EACH ITERATION.
# A PROPER TEST OF PARALLELIZATION WILL HAPPEN WHEN I CHECK THIS ON CUDA ONLY. HENCE, TOMO, ON 
# O2, TEST THE PARALLELIZATION BEHAVIOUR OF TORCH BY DOING SAME THING AS ABOVE, BUT ON CUDA AS 
# THE DEVICE.

# using cuda, with parallelization
starttime = time.time()
for _ in range(10000):
	a = torch.randn(size = (2,2), device = torch.device('cuda'))
	b = torch.mm(a, a)
	torch.cpu.synchronize()
torch.cuda.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.794

# using cuda, with parallelization, but after each iteration, we synchronize the devices
starttime = time.time()
for _ in range(10000):
	a = torch.randn(size = (2,2), device = torch.device('cuda'))
	b = torch.mm(a, a)
	torch.cuda.synchronize()
torch.cuda.synchronize()
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.875

# BELOW, USING TWO GPUS AND IN THE CASE OF PARALLELIZATION, SYNCHRONIZING THE OTHER GPU INSTEAD 
# AT THE END OF EACH ITERATION TO REMOVE THE BIAS DUE TO THE TIME TAKEN TO RUN THE SYNCHORNIZATION
# COMMAND ITSELF.

gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:1')

# using cuda, with parallelization, but after each iteration, we synchronize the devices
starttime = time.time()
for _ in range(1000):
	a = torch.randn(size = (100,100), device = gpu0)
	b = torch.mm(a, a)
	torch.cuda.synchronize(device = gpu1)
torch.cuda.synchronize(device = gpu0)
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.018

# making the operations serial by waiting for all kernels on all streams on the cuda0 device 
# to be completed at the end of each iteration
starttime = time.time()
for _ in range(1000):
	a = torch.randn(size = (100,100), device = gpu0)
	b = torch.mm(a, a)
	torch.cuda.synchronize(device = gpu0)
torch.cuda.synchronize(device = gpu0)
endtime = time.time()
print(round(endtime - starttime, 3)) # 0.021

# So here we see a nice, consistent difference of 0.003 seconds. This difference is due to the 
# fact that we are waiting for the kernels in the GPU0 to complete at the end of each iteration
# in the second case, which is leading to some idle time for the frontend, and that is where 
# those 0.003 seconds are going.




































