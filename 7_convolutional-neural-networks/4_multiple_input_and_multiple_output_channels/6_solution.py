# When we are performing convolution on a strip using the kernel strip we are basically performing
# one (first case) or Δ (second case) number of dot products. Hence, its simply a matrix 
# multiplication. Its preferable to perform a bigger matrix multiplication to be able to perform
# more computation in one clock-cycle of the CPU. The vector units (SIMD) of the CPU are specialized 
# in dealing with vectorized operations and allow us to make use of the computational power of the 
# CPU with higher efficiency. Hence, computationally, the second case helps.
# Also, if we notice, there is redundancy in matrix contents for the different matrix 
# multiplications. Hence, loading a bigger strip from the input image helps because we are not 
# loading proportionally large amount of data from the memory. We are loading lesser, which allows
# us to have lesser individual transfers of data from the memory and more efficiency in that too, 
# which also helps in reducing the time taken. 
# We should keep Δ as big as the SIMD performance doesn't reduce much (meaning the vector units)
# are able to hold the tensors on which the matrix multiplication is computed, and also such that
# the caches are not overwhelmed by the amount of data transferred in one go to them. If they are
# overwhelmed, then there will be more cache misses, and getting the data in the first place will
# becoming the rate limiting step. 