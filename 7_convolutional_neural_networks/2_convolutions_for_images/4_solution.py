# To achieve this, we will have to use torch batch matrix multiplication. 
# Assume input matrix has shape (n,n), kernel has shape (m,m)

# If we carefully see, we will observe that the kernel takes (n-1,n-1) positions in the cross 
# correlation process. And corresponding to each position, there is a dot-product happening.
# We can arrange the input vector into a shape of (n-1,n-1,m*m), where each each row of each height
# corresponds to the entries in original input vector that overlapped with the kernel in a 
# particular configuration. The kernel vector can be transformed into a vector of shape (n-1,m*m,1), 
# where each height's only column holds the flattened kernel. 

# Hence, when we do the batch matrix multiplication, we get a matrix of shape (n-1,n-1,1) (which 
# can be reshaped into shape (n-1,n-1)) that holds the result of the cross correlation. 







