# Yes, thats true that different devices, like say, CPUs and GPUs have different compute 
# capabilities. Hence, ideally, we want to divide the total task between these devices such that
# they are playing at their strengths. That is exactly what we try to do by making the CPU as host
# whose job is to submit tasks to the different GPUs, which do the actual computation. CPU is 
# more general processing capable, hence is suitable for the task of coordination and distributing
# tasks, whereas GPU is great at actual computations, hence can handle that part. 
# It is worth the effort if the gains of distribution are more than the communication overhead.
# Generally, with most decently sized models, it is worth to split up the job between CPU and GPU
# as described above. Maybe for super small models, it might not be worth it, and would be faster 
# to do everything on the GPU itself.