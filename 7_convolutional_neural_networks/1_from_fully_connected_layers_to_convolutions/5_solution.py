# dealing with boundary cases in convolutions leads to many possibilities. Either we can ignore 
# some possible 'centerpoints' of the kernel if it leads to some of the neighbors being absent.
# Else, we can add necessary padding to each border with a neutral value (0 maybe), so that we 
# can still have an output with the same size as the input image. Hence, according to what 
# approach was used, we will either completely miss out on the presence of an object at the 
# boundary of an image, or we will not be able to recognize it because of insufficient information 
# about it from the image (and effects of padding etc).