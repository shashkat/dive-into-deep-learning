# This is easy to intuitively understand that the kernel size would be atleast (d+1)x(d+1). This 
# is because in the finite difference approximation of order d, we would have to access the term 
# for f(x+dh) where h is the smallest increment in the approximation. And to access that term, we 
# would need a kernel which would be able to reach that hence be atleast of dimension (d+1).