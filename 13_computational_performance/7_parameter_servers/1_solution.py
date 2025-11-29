# We can make use of the bidirectionality of data-transfer-ability by splitting the gradients at 
# each GPU into n_gpu*2 parts instead of n_gpu parts. Then, for the ith GPU, the ith part of its
# gradients will also be split into two parts (say A and B), and the Ath part will go through the
# GPUs in the clockwise direction and will eventually get aggregated on the GPU to the left of 
# GPUi, and the B part will go through the GPUs in counter-clockwise direction and will eventually
# get aggregated on the GPU to the right of GPUi. This way, the data transferred in each step will
# be half of what we discussed in the chapter and the time taken would be: (n-1)/2n ~ 1/2, meaning
# half of before (still O(1) though).