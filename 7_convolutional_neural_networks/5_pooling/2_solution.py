# If the kernel was to return the max value that lies in the current window position, it would 
# maybe need to have values something like 1 corresponding to the value that overlaps with the 
# maximum value in the input tensor for that window position, and 0 for all other kernel entries.
# It could also have something like a linear operation on all the entries such that the result 
# would look like the max value in the current window position. However, no matter how it computes
# the "maximum" for current window position, it would encode where the maximum was to begin with.
# If the maximum would move to another position within the window, the same linear operation
# wouldn't necessarily yield the maximum still. It would likely yield the value in the position 
# where the maximum was before. Hence, it is not possible to use a convolution for a maximum 
# pooling operation.