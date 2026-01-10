# Its straightforward to think about this using the worded-definition: "convolution means measuring 
# the overlap when one function is flipped and shifted by x". 
# If suppose both f and g are functions with 2d input. When we do f*g, that means we are flipping
# f (in both the dimensions of input, so that each input vector is flipped), and shifting it such
# that its origin now overlaps with location of x (2d vector). Now, we compute the overlap between
# the two curves (to get the value of f*g for current value of x). If we think about what coordinate
# from the domain of f now coincides with origin of g, we can see that it would be x, as f is
# flipped and its origin is made to coincide with x in g's domain. If we would have done g*f,
# then we would have flipped g and shifted to match its origin with x in f's domain. And automatically
# that would lead to x in g's domain coinciding with origin in f. Hence, we can see that the same
# points coincide when we do f*g or g*f for a given x. Hence, f*g(x) = g*f(x) for all x. Which means
# that convolution is symmetric.












