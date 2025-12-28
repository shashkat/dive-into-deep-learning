import torch 
import pandas as pd

# print entries from dir function of an object, which have a particular substring
def DirSubsetToSubstring(obj, substr):
	entries_with_substr = []
	for entry in dir(obj):
		if substr in entry:
			entries_with_substr.append(entry)
	return entries_with_substr

# create a tensor and make its requires_grad to be true
a = torch.ones(5)
a.requires_grad = True

# derive another tensor from a
b = 2*a
b.retain_grad()   # Since b is non-leaf and it's grad will be destroyed otherwise.

# create a final (root) tensor and call .backward() from it
c = b.mean()
c.backward()

# print the gradients of a and b
print(a.grad, b.grad)
# In [13]: print(a.grad, b.grad)
# tensor([0.4000, 0.4000, 0.4000, 0.4000, 0.4000])
# tensor([0.2000, 0.2000, 0.2000, 0.2000, 0.2000])

############################################################################
# Redo everything with a simple hook that prints gradient of b upon the backward propagation 
# going through it
############################################################################

# create a tensor and make its requires_grad to be true
a = torch.ones(5)
a.requires_grad = True

# derive another tensor from a
b = 2*a
b.retain_grad()   # Since b is non-leaf and it's grad will be destroyed otherwise.

# register to b, a backward hook (as register_hook function is used for backward hooks only), 
# which prints the gradient of it as the backpropagation goes through it (meaning computes 
# derivate of root wrt b, also called gradient of b)
b.register_hook(lambda x: print(x))

# just checking which attributes/methods of b have 'hook' in them
DirSubsetToSubstring(b, 'hook')

# create a final (root) tensor and call .backward() from it
c = b.mean()
c.backward()
# so as we see, upon calling backward on c, the gradient (derivative of c wrt b) of b was printed

# print the gradients of a and b
print(a.grad, b.grad)

# Conclusion: we can see that upon calling backward on c, the gradient (derivative of c wrt b) 
# of b was printed

############################################################################
# Checking significance of making retains_grad attribute of b True
############################################################################

# create a tensor and make its requires_grad to be true
a = torch.ones(5)
a.requires_grad = True

# derive another tensor from a
b = 2*a
# b.retain_grad() 

# register to b, a backward hook (as register_hook function is used for backward hooks only), 
# which prints the gradient of it as the backpropagation goes through it (meaning computes 
# derivate of root wrt b, also called gradient of b)
b.register_hook(lambda x: print(x))

# create a final (root) tensor and call .backward() from it
c = b.mean()
c.backward()
# so as we see, upon calling backward on c, the gradient (derivative of c wrt b) of b was printed

# print the gradients of a and b
print(a.grad, b.grad)

# Conclusion: # the register hook works as it should even without making the retains_grad 
# attribute of b to be True. Its just that we wont be able to print it at the end of the backward 
# call like we are doing now, along with a's grad.

############################################################################
# Registering a hook function to b, which always returns a fixed tensor (making the grad of b 
# to be that fixed tensor irrespective of how c was obtained from b)
############################################################################

# create a tensor and make its requires_grad to be true
a = torch.ones(5)
a.requires_grad = True

# derive another tensor from a
b = 2*a
b.retain_grad()

# register to b, a backward hook (as register_hook function is used for backward hooks only), 
# which always returns a fixed tensor.
b.register_hook(lambda x: torch.tensor([1,2,3,4,5], dtype = torch.float))

# create a final (root) tensor and call .backward() from it
c = b.mean()
c.backward()

# print the gradients of a and b
print(a.grad, b.grad)

# Conclusion: the gradient of b (derivative of c wrt b) was made to be the supplied tensor 
# irrespective of how c was computed from b, and that also changed the gradient of a as it is 
# determined from that of b.


#### temp
# just checking which attributes/methods of b have 'hook' in them
DirSubsetToSubstring(b, 'grad')
b.retains_grad




# Redo the experiment but with a hook that multiplies b's grad by 2.
a = torch.ones(5)

a.requires_grad = True

b = 2*a

b.retain_grad()

b.register_hook(lambda x: print(x))  

b.mean().backward() 


print(a.grad, b.grad)