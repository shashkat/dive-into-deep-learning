# here, I try to investigate how many cpu cores are available (as I am running code in modal 
# to maintain uniformity), how many of threads are used by default, how I can change them, and
# how the tracing will look like when we vary the number of threads used.

import torch
import os
os.cpu_count() # okay, 17 cpus.

# get number of threads by default
torch.get_num_threads() # 1 by default

# set number of threads used for intraop parallelism on CPU to be 10 instead of the default 1.
torch.set_num_threads(10)

### now lets do some basic operations on cpu and make the trace

tensor_on_cpu = torch.randn(size = (10,10), device = 'cpu')
# function that involves computation on cpu
def ComputationOnCPU(n_iter = 100):
	for i in range(n_iter):
		torch.mm(input = tensor_on_cpu, mat2 = tensor_on_cpu)
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

# So upon doing this, I didn't observe any trace for the other threads apart from the main 
# thread (that actually goes through each command to execute). I asked perplexity and according
# to it, this is because pytorch profiler doesn't have the ability to show the profile of 
# threads on cpu other than the main thread, even though intra-op parallelization is happening 
# here (https://www.perplexity.ai/search/i-am-trying-to-profile-the-tra-nGU9M0_PSMC0K5vWfo1m_Q#1).









