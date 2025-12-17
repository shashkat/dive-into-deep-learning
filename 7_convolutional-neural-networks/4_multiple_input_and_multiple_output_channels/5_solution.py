# Assume input matrix has shape (ci,nh,nw), kernel has shape (co,ci,kh,kw).

# one channel of the output will have shape (nh-kh-1 x nw-kw-1). We can reshape the input to 
# shape ((nh-kh-1)*(nw-kw-1) x ci*kh*kw) and the kernel into shape (ci*kh*kw x co). Here, each 
# row in reshaped input corresponds to one overlap position between the 2d kernel shape and the 
# image. For a given overlap position, the tensor values are taken from all the input channels 
# and flattened into a single dimensional vector, with first kh*kw entries from the first input 
# channel for that overlap, next kh*kw entries from the second input channel for that overlap and 
# so on. Each column in the reshaped kernel corresponds to all the kernels corresponding to 
# one output channel flattened into one vector. 
# Hence, when we mulitply these two matrices, we will get a matrix of shape 
# ((nh-kh-1)*(nw-kw-1) x c0), which can be reshaped to shape (co, (nh-kh-1), (nw-kw-1)), which is 
# our final result of the convolution.
