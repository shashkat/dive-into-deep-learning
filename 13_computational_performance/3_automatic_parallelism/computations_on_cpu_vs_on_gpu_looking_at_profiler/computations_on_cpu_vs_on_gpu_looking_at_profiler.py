# here, I try to do some computations on cpu vs gpu, and make their torch profiler traces, so 
# that I can compare them. I was interested in seeing that if the computations on GPU are 
# asynchronous, i.e. the CPU thread (host) just sends them to GPU for computation, then how does 
# the time taken by CPU when it reaches a function call look like when that function is worked 
# on on GPU vs on CPU. I expect that the time (taken by CPU) for functions of varying complexity
# executed on GPU should be almost same as the CPU just has to send the command to GPU, but if 
# they were computed on the CPU itself, then the time taken would vary. 

import torch

### 100 SIMPLE COMPUTATIONS

# declare two tensors, one on cpu and another on gpu
tensor_on_cpu = torch.randn(size = (10,10), device = 'cpu')
tensor_on_gpu = torch.randn(size = (10,10), device = 'cuda')

# function that involves computation on cpu
def ComputationOnCPU(n_iter = 100):
	for i in range(n_iter):
		torch.mm(input = tensor_on_cpu, mat2 = tensor_on_cpu)
	return

def ComputationOnGPU(n_iter = 100):
	for i in range(n_iter):
		torch.mm(input = tensor_on_gpu, mat2 = tensor_on_gpu)
	return

# now, lets profile the computations on cpu first and observe their trace
with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	schedule=torch.profiler.schedule(
		wait=1, warmup=1, active=1, repeat=1, skip_first=1 # this setup requires 4 iterations of whatever is being profiled
	)
) as prof:
	for step_idx in range(4):
		# call the function to profile 
		ComputationOnCPU()
		prof.step()
prof.export_chrome_trace('my_vol/computation_on_cpu.json')

# now lets profile the computations on gpu and observe their trace
with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	schedule=torch.profiler.schedule(
		wait=1, warmup=1, active=1, repeat=1, skip_first=1 # this setup requires 4 iterations of whatever is being profiled
	)
) as prof:
	for step_idx in range(4):
		# call the function to profile 
		ComputationOnGPU()
		prof.step()
prof.export_chrome_trace('my_vol/computation_on_gpu.json')

### 10 COMPLICATED COMPUTATIONS (EACH OF WHICH MIGHT TAKE LOT MORE TIME ON CPU THAN GPU)

# declare two tensors, one on cpu and another on gpu
tensor_on_cpu = torch.randn(size = (1000,1000), device = 'cpu')
tensor_on_gpu = torch.randn(size = (1000,1000), device = 'cuda')

# function that involves computation on cpu
def ComputationOnCPU(n_iter = 10):
	for i in range(n_iter):
		torch.mm(input = tensor_on_cpu, mat2 = tensor_on_cpu)
	return

def ComputationOnGPU(n_iter = 10):
	for i in range(n_iter):
		torch.mm(input = tensor_on_gpu, mat2 = tensor_on_gpu)
	return

# now, lets profile the computations on cpu first and observe their trace
with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	schedule=torch.profiler.schedule(
		wait=1, warmup=1, active=1, repeat=1, skip_first=1 # this setup requires 4 iterations of whatever is being profiled
	)
) as prof:
	for step_idx in range(4):
		# call the function to profile 
		ComputationOnCPU()
		prof.step()
prof.export_chrome_trace('my_vol/computation_on_cpu.json')

# now lets profile the computations on gpu and observe their trace
with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	schedule=torch.profiler.schedule(
		wait=1, warmup=1, active=1, repeat=1, skip_first=1 # this setup requires 4 iterations of whatever is being profiled
	)
) as prof:
	for step_idx in range(4):
		# call the function to profile 
		ComputationOnGPU(n_iter = 10)
		torch.cuda.synchronize()
		prof.step()
prof.export_chrome_trace('my_vol/computation_on_gpu.json')










