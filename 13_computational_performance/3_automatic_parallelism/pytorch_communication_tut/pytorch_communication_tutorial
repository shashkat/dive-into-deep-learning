# here, I run and try to understand better, the code demonstrating communication between
# cpu and gpu in pytorch from the documentation (https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)


import time
import numpy


# they are trying to demonstrate conditions which must be met for a transfer from cpu to gpu
# to be non-blocking for the gpu. The non-blocking argument of the .to() method dictates if the
# transfer will be non-blocking for the source (cpu). Hence, how to make it non-blocking for the
# target (gpu), is shown below.
import contextlib
import torch
from torch.cuda import Stream

s = Stream()

torch.manual_seed(42)
t1_cpu_pinned = torch.randn(1024**2 * 5, pin_memory=True)
t2_cpu_paged = torch.randn(1024**2 * 5, pin_memory=False)
t3_cuda = torch.randn(1024**2 * 5, device="cuda:0")

assert torch.cuda.is_available()
# store in the device variable, the cuda device we are currently using. The second argument 
# here indicates the index of the cuda device we are indicating
device = torch.device("cuda", torch.cuda.current_device())

# The function we want to profile
def inner(pinned: bool, streamed: bool):
	# pinned = False
	# streamed = False

	# if streamed, then we work within the context of stream s. Else, we work in no context.
    with torch.cuda.stream(s) if streamed else contextlib.nullcontext():
        if pinned:
            t1_cuda = t1_cpu_pinned.to(device, non_blocking=True)
        else:
            t2_cuda = t2_cpu_paged.to(device, non_blocking=True)
        t_star_cuda_h2d_event = s.record_event() # create an event in stream s, which can be used later for synchronization or timing purposes.
    # This operation can be executed during the CPU to GPU copy if and only if the tensor is 
    # pinned and the copy is done in the other stream
    t3_cuda_mul = t3_cuda * t3_cuda * t3_cuda
    t3_cuda_h2d_event = torch.cuda.current_stream().record_event()
    t_star_cuda_h2d_event.synchronize() # make the cpu thread wait till this event is completed
    t3_cuda_h2d_event.synchronize() # make the cpu thread wait till this event is completed

# Our profiler: profiles the `inner` function and stores the results in a .json file
def benchmark_with_profiler(pinned, streamed) -> None:
	# configure the profiler to force synchronizations in the GPU for each operation
    torch._C._profiler._set_cuda_sync_enabled_val(True)
    wait, warmup, active = 1, 1, 2 # the arguments for profiler schedule
    num_steps = wait + warmup + active
    rank = 0
    # we use the torch profiler's profile entrypoint as a context manager, and indicate which 
    # activities to profile, and 
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU, # profile the activity of cpu
            torch.profiler.ProfilerActivity.CUDA, # profile the activity of cuda (gpu)
        ],
        # specify the number of wait, warmup, active steps in the schedule.
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            inner(streamed=streamed, pinned=pinned)
            if rank is None or rank == 0:
                prof.step()
    prof.export_chrome_trace(f"trace_streamed{int(streamed)}_pinned{int(pinned)}.json")

# do the profiling of the inner function.
benchmark_with_profiler(streamed=False, pinned=False)
benchmark_with_profiler(streamed=True, pinned=False)
benchmark_with_profiler(streamed=False, pinned=True)
benchmark_with_profiler(streamed=True, pinned=True)

### here trying some ways of copying and accessing which can render the accessed data useless
import torch

DELAY = 100000000
try:
    i = -1
    for i in range(100):
    	# i = 0
        # Create a tensor in pin-memory
        cpu_tensor = torch.ones(1024, 1024, pin_memory=True)
        torch.cuda.synchronize()
        # Send the tensor to CUDA
        cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
        torch.cuda._sleep(DELAY)
        # Corrupt the original tensor
        cpu_tensor.zero_()
        assert (cuda_tensor == 1).all()
    print("No test failed with non_blocking and pinned tensor")
except AssertionError:
    print(f"{i}th test failed with non_blocking and pinned tensor. Skipping remaining tests")

# making trace of the above code to better understand what is happening
cpu_tensor = torch.ones(1024, 1024)
torch.cuda.synchronize()
with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	schedule = torch.profiler.schedule(
		wait = 1, warmup = 1, active = 1, repeat = 1, skip_first = 1
		)
) as prof:
	for step_idx in range(4):
		# Send the tensor to CUDA
		cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
        torch.cuda._sleep(DELAY)
        # Corrupt the original tensor
        cpu_tensor.zero_()
		torch.cuda.synchronize()
		prof.step()
	prof.export_chrome_trace('my_vol/trace_corrupt_test.json')







│ trace_streamed1_pinned1.json │ file │ 2025-10-14 21:27 EDT │ 18.9 KiB │
│ trace_streamed1_pinned0.json │ file │ 2025-10-14 21:27 EDT │ 18.8 KiB │
│ trace_streamed0_pinned1.json │ file │ 2025-10-14 21:27 EDT │ 18.7 KiB │
│ trace_streamed0_pinned0.json │ file │ 2025-10-14 21:19 EDT │ 18.4 KiB │

trace_streamed1_pinned1.json trace_streamed1_pinned0.json trace_streamed0_pinned1.json trace_streamed0_pinned0.json

@app.function(volumes={"/my_vol": modal.Volume.from_name("vol")})

modal volume put vol /Users/shashankkatiyar/Documents/learning_ml/modal_test/outputs /app

import modal

image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11").pip_install("numpy", "torch", "pandas", "scipy", "ipython").add_lo
cal_dir(
        local_path="/Users/shashankkatiyar/Documents/learning_ml/modal_test",
        remote_path="/app"
)

app = modal.App("test_app", image=image)

@app.function(gpu="T4:1")
def main():
        print('Entered the function in python file!')













