# here, I try to test how I can transfer data from 1 gpu to another with it being non-blocking 
# on both the sender gpu and receiver gpu. This is especially different from the case of cpu 
# to gpu because the memory should be pinned in cpu for it to be nonblocking on the receiver, 
# but when its from gpu to gpu, its not possible as there is no concept of pinned memory in gpu.

# NOTE: One needs to have access to atleast two GPU devices for this code

import contextlib
import re
import torch
from torch.cuda import Stream
import time
import numpy

# make secondary streams on both gpus, which will be used for the transfer
s0 = Stream(device = torch.device('cuda', 0))
s1 = Stream(device = torch.device('cuda', 1))

torch.manual_seed(42)
A = torch.randn(1024**2 * 10, device="cuda:0")
B = torch.randn(1024**2 * 10, device="cuda:1")

# store in device variable, the cuda device onto which we will be making the transfer (receiver)
device = torch.device("cuda", 1)

# The function we want to profile.
def inner(sender_streamed = False, receiver_streamed = False, non_blocking = True):
	# sender_streamed = False
	# receiver_streamed = True
	# non_blocking = True

	# according to values of arguments, we do the transfer in appropriate context
	if sender_streamed and receiver_streamed:
		with torch.cuda.stream(s0), torch.cuda.stream(s1):
			A_transferred = A.to(device, non_blocking = non_blocking) # having a separate argument for non_blocking as want to test if non-blocking effects cpu behaviour only, or effects gpus also in any direct way
        	secondary_stream0_event = s0.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu0 (s0), which can be used later for synchronization or timing purposes.
        	secondary_stream1_event = s1.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu1 (s1), which can be used later for synchronization or timing purposes.
        	# the purpose of creating these events is that I can synchronize them with cpu at the
        	# end of this function and ensure that profiling is done till both of them have been 
        	# reached which means that the transfer is done
    if sender_streamed and not receiver_streamed:
    	with torch.cuda.stream(s0):
			A_transferred = A.to(device, non_blocking = non_blocking) # having a separate argument for non_blocking as want to test if non-blocking effects cpu behaviour only, or effects gpus also in any direct way
        	secondary_stream0_event = s0.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu0 (s0), which can be used later for synchronization or timing purposes.
        	secondary_stream1_event = s1.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu1 (s1), which can be used later for synchronization or timing purposes.
        	# it doesn't really harm to have these two events in the two streams. If there was nothing queued on the stream, then the event will be reached superfast and there will be no time loss when we synchronize the cpu wrt it
    if not sender_streamed and receiver_streamed:
    	with torch.cuda.stream(s1):
			A_transferred = A.to(device, non_blocking = non_blocking) # having a separate argument for non_blocking as want to test if non-blocking effects cpu behaviour only, or effects gpus also in any direct way
        	secondary_stream0_event = s0.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu0 (s0), which can be used later for synchronization or timing purposes.
        	secondary_stream1_event = s1.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu1 (s1), which can be used later for synchronization or timing purposes.
        	# it doesn't really harm to have these two events in the two streams. If there was nothing queued on the stream, then the event will be reached superfast and there will be no time loss when we synchronize the cpu wrt it
    if not sender_streamed and not receiver_streamed:
    	with contextlib.nullcontext():
			A_transferred = A.to(device, non_blocking = non_blocking) # having a separate argument for non_blocking as want to test if non-blocking effects cpu behaviour only, or effects gpus also in any direct way
        	secondary_stream0_event = s0.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu0 (s0), which can be used later for synchronization or timing purposes.
        	secondary_stream1_event = s1.record_event() # create an event and record it (imagine record here like a screenshot, as it corresponds to a single timepoint) in secondary stream on gpu1 (s1), which can be used later for synchronization or timing purposes.
        	# it doesn't really harm to have these two events in the two streams. If there was nothing queued on the stream, then the event will be reached superfast and there will be no time loss when we synchronize the cpu wrt it

    # This operation can be executed during the GPU to GPU copy if and only if the 
    # the copy is done in the context of secondary streams in both gpus
    B_multiplied = B * B * B * B
    base_stream_event = torch.cuda.current_stream(device = device).record_event() # record an event in the default stream of the receiver gpu
    secondary_stream0_event.synchronize() # make the cpu thread wait till this event is completed
    secondary_stream1_event.synchronize() # make the cpu thread wait till this event is completed
    base_stream_event.synchronize() # make the cpu thread wait till this event is completed (this synchronization marks completion of multiplication task)

# Our profiler: profiles the `inner` function and stores the results in a .json file
def benchmark_with_profiler(sender_streamed = False, receiver_streamed = False, non_blocking = True) -> None:
	# configure the profiler to force synchronizations in the GPU for each operation
    # torch._C._profiler._set_cuda_sync_enabled_val(False) # True forces a cudaDeviceSynchronize after each 
    # kernel is launched to make sure the measurement includes kernel completion. Basically what this command 
    # does is allows us to accurately measure the completion time of kernel launch by cpu (https://www.perplexity.ai/search/this-is-the-pytorch-profiler-t-BfJs9jbvTjGvJ_7y9AhshQ#2). 
    # However later, when realized that in the profile trace, more details always appeared when the ipython 
    # instance was fresh, tried this with False, and it looked basically same as with True. Hence, I think 
    # this is not required to be played with
    
    wait, warmup, active = 1, 1, 2 # the arguments for profiler schedule
    num_steps = wait + warmup + active
    rank = 0
    # we use the torch profiler's profile entrypoint as a context manager, and indicate which 
    # activities to profile, and how to schedule them.
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
            inner(sender_streamed, receiver_streamed, non_blocking)
            if rank is None or rank == 0:
                prof.step()
    prof.export_chrome_trace(f"trace_senderstreamed{int(sender_streamed)}_receiverstreamed{int(receiver_streamed)}_nonblocking{int(non_blocking)}.json")

# do the profiling of the inner function.
for i in [True, False]:
	for j in [True, False]:
		for k in [True, False]:
			benchmark_with_profiler(sender_streamed=i, receiver_streamed=j, non_blocking=k)			





### old stuff below

benchmark_with_profiler(streamed=True)


# figuring out using events to be able to capture the time taken for different tasks, even 
# when the profiler trace displays less info.


benchmark_with_profiler(streamed=True, itr = 0)
benchmark_with_profiler(streamed=True, itr = 1)
benchmark_with_profiler(streamed=True, itr = 2)
benchmark_with_profiler(streamed=True, itr = 3)
benchmark_with_profiler(streamed=True, itr = 4)
benchmark_with_profiler(streamed=True, itr = 5)

# HERE... THERE DOESN'T SEEM TO BE APPEARING THE GPU COMPUTATIONS ON THE PROFILER. I HAVE NOT 
# CLUE WHY. HOWEVER, WHEN THE STREAM S WAS IN GPU:0, THEN THE MULTIPLICATIONS WERE APPEARING, 
# WHICH WAS LIKELY RANDOMLY, AS WHEN RETRIED KEEPING S ON DEVICE 0, IT DIDN'T WORK AGAIN.
# I NEED TO DETERMINE WHEN THE OTHER STREAMS APPEAR IN THE PROFILING (BASICALLY WHEN THE 
# PROFILING IS MORE DETAILED AND WHEN IT IS NOT, AS THAT SEEMS TO CHANGE RANDOMLY), AND WHEN 
# THEY DONT. 
# THOUGHT ABOUT THE USEFULNESS OF torch._C._profiler._set_cuda_sync_enabled_val(True) AND IT
# SEEMS TO BE USEFUL AS IT TELLS ACCURATE TIMES FOR KERNEL LAUNCHES BY CPU. BUT STILL NEED TO 
# FIGURE OUT HOW TO MAKE THE GPU PROCESSES VISIBLE. WAS LOOKING AT PERPLEXITY ANSWER ABOUT 
# SOMETHING RELATED TO GPU TASK EASINESS WHICH MIGHT MAKE THINGS ABSENT FROM THE TRACE.


benchmark_with_profiler(streamed=True, pinned=False)
benchmark_with_profiler(streamed=False, pinned=True)
benchmark_with_profiler(streamed=True, pinned=True)





