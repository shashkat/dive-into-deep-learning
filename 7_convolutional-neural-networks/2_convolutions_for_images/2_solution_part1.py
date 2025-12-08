# 1

# Thought a lot about this. Couldn't come up with a reasonable solution myself. Then asked 
# perplexity (https://www.perplexity.ai/search/how-would-you-approach-this-qu-uCsAyEGNRi.y9XSzXyvVuQ#0)
# Turns out this matrix is the kernel required to detect edges perpendicular to vector v (v1,v2).
# It is basically a finite difference approximation of directional gradient in the direction of v.
# [0, 	v2]
# [v1, 	-(v1+v2)]
# As we can see, if v1 is 0 and v2 is 1, meaning vertical vector, then it becomes a horizontal 
# edge detector, and vice versa.
