# Yes, it is possible to allow asynchronous communication (while computation is still going on),
# by aggregating the gradients as they are computed from front to back. This is what DDP 
# (Distributed Data Parallel) tries to achieve in pytorch. We can simply divide all the gradients
# to be computed into n groups, and as soon as a particular group is computed (front to back),
# it can be aggregated by the GPUs while they are computing the group just before that.
# This will lead to quite an improvement in performance, especially in case of big models, when
# both computations and communication individually take a lot of time.