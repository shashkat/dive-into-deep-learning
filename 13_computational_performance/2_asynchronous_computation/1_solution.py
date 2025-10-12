import torch
import numpy
from d2l import torch as d2l

# we can test if asynchrony is happening on the cpu by measuring the time it takes to compute 
# some operations "possibly in an asynchronized way", and some operations in a necessarily 
# synchronized way, and compare the time it takes to compute them. Its a bit difficult to do this 
# in pytorch for cpu, the way I tested the same for matrix multiplications in gpu devices 
# (https://shashkat.github.io//posts/asynchronous-computations-gpu/), because there is no effect
# of torch.cpu.synchronize(), simply because they know that there is no asynchronicity in CPU.
# But if we want to validate it, we need a different approach. 

# Hence, I think it is better that we compare the times taken to do the same matrix multiplications
# using numpy and pytorch on cpu. Since numpy for sure doesn't do any asynchronous computations,
# that is our control case. We assume that there are no differences in time taken for matrix
# multiplications by pytorch and numpy if we ignore asynchronicity.

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)
# appx 0.6 sec

device = torch.device('cpu')
with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
# appx 0.27 sec

# The time difference is due to the difference in the way matrix multiplication is performed by 
# numpy and torch. Here, torch seems to be faster doing matrix multiplication on GPU. But if numpy's
# dynamics of doing matrix multiplication were same as pytorch, the time taken would have been 
# similar. Overall, operations by pytorch in CPU are synchronized.














