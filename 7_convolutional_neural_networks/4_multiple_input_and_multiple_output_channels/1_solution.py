# 1
# Yes we can always combine two convolution operations (without a nonlinearity between) into 
# one convolution. Suppose the first convolution (performed on image I0) gives an image I1, and 
# second convolution gives image I2. 
# One step of the second convolution would look like: k2_11 * I1_11 + k2_12 * I1_12 and so on..
# Here, we can replace the I1_11 and similar terms with what computation was performed on elements
# of I0 in the first place in the first convolution. Hence, it would look something like: 
# k2_11 * (k1_11*I0_11 + k1_12*I0_12 + ...) + k2_12 * (k1_11*I0_12 + k1_12*I0_13 + ...). 
# We can open the brackets and get something like: 
# k2_11*k1_11*I0_11 + k2_11*k1_12*I0_12 + ... + k2_12*k1_11*I0_12 + k2_12*k1_12*I0_13 + ...
# Hence, we can see that its just a different set of numbers being multiplied to elements of I0,
# and being added, which is nothing but a convolution, just with a different kernel.

# 2
# The dimensionality of the equivalent single convolution is (k1h*k2h, k1w*k2w), where k1 
# corresponds to the first kernel and k2 corresponds to the second kernel.

# 3
# If we ignore kernels with just one element, then we can not always break down a convolution 
# into two smaller convolutions. This can be proven simply by making the point that if the kernel 
# has a size which is prime (say 3), then the factor kernels cannot have a whole number size 
# in the same dimension, hence it cannot be broken down into two factor kernels.


