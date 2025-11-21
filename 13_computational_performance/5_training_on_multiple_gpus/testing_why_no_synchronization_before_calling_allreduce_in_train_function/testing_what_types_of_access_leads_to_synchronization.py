# want to test that if something is to be executed in different device, will the main thread 
# become free to execute other commands (in ipython), even before the other device is done?


# to test, I create a tensor on gpu. I print it and time how long it takes to print the contents.
# Then, I submit commands from the main thread to perform heavy computations on the tensor on 
# gpu and time that submission. Then, I do the same thing but also print the contents of the 
# tensor just after submitting the computation. 
# My hypothesis is that printing the tensor initially will take very less time. Then just the 
# submission of the computation will also take very less time. But finally, the print 
# command after the computation will take a lot more time, as that actually indicates the 
# computation time.

import torch
from torch import nn
import time

temp = torch.randn(size = (10000, 10000), device = torch.device('cuda'))

# print the tensor simply
starttime = time.time()
print(temp)
endtime = time.time()
round(endtime - starttime, 4) # 0.003

# perform big computation on tensor (just submitting the task by main thread shouldnt take much 
# time)
starttime = time.time()
torch.mm(temp, temp) # this command doesn't count as an access of the results, hence the main thread just submits it and moves ahead.
endtime = time.time()
round(endtime - starttime, 4) # 0.0003

# perform same big computation on tensor, but also print the contents, which require the main 
# thread on cpu to wait for the computation to be complete before reaching endtime = time.time()
starttime = time.time()
print(torch.mm(temp, temp)) # this counts as access of the results of the computation, hence the main thread waits for it to be done
endtime = time.time()
round(endtime - starttime, 4) # 0.56

# try the same thing as above, but print the contents of something else that resides in the gpu. 
# Lets see if its the whole GPU that gets blocked or just the part of memory which is involved 
# in some computation currently. My guess is that the whole GPU is gonna be blocked.
temp2 = torch.randn(size = (10000, 10000), device = torch.device('cuda'))
starttime = time.time()
torch.mm(temp, temp) # this counts as access of the results of the computation, hence the main thread waits for it to be done
print(temp2)
endtime = time.time()
round(endtime - starttime, 4) # 0.6

# Hence, we can see that the print command forces synchronization of the gpu, and its not just 
# printing the thing on which computation is currently being done. It is anything that resides 
# on that GPU.













