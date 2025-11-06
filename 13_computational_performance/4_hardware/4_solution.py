# So when we talk about having multiple memory channels between the RAM and the CPU (memory 
# controller in it), in order to have full usage of all the channels for data transfer (and 
# hence high bandwidth), it is better to have the data interleaved between the different modules
# of RAM that the channels are connected to. This means that sequential parts of data are 
# distributed across the RAM modules connected to the different memory channels, so that when we 
# access data, each memory channel has some work to do, and we utilize the full bandwidth. 

# First of all, definition of thread is: a sequence of tasks. It is a software thing and not 
# hardware. When we talk about having multiple threads, and distributing any task between them, 
# it is generally intuitive to understand that we don't want the threads to be trying to access
# the same data, as then some will have to wait while the others access and possibly modify that 
# data. Hence, for optimally using threads, we need the data to be divided into chunks, and each 
# chunk has contiguous parts of data, and is accessed by one thread.

# The two data organization requirements are different. Hence, it is good to have the data 
# interleaved between the different RAM modules, but there should be chunks of data in each 
# module which are contiguous. Hence, we ensure that the threads don't compete for same piece 
# of memory as well as ensure that the momery channels are utilized well.