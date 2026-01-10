# Using a kernel with size = 0 in convolution means to multiply just the center pixel with a 
# number and move on to the next one. We don't see any of its neighbors. Hence, if we look
# at one channel of the input image, we are basically multiplying all the pixels with the same 
# number. To obtain one channel of the output image, we multiply each channel of the input image
# with a fixed number each (but different among the channels), and add them all to get a single
# image, and add a fixed number to all the pixels finally (which is the bias term). 

# One can view this in a different way too. To obtain one pixel of one channel of the output image,
# we multiply the same positioned pixel from all channels in the input image with different 
# numbers, add them up, and add a bias term. Similarly, we mulitply all the channels of the same 
# pixel with another set of number and add them up (and a bias) to get the value of the same pixel
# but another channel in the output image. Then, we move on to the next pixel in input, and do the 
# same and so on. Hence, in this perspective, each pixel is a datapoint, and we are passing it 
# through a single layer of an MLP to get the different channel values for that pixel in the 
# output image




