import torch
from torch import nn

# function to compute 2d cross correlation. Returns a matrix of appropriate shape after applying 
# cross-correlation on matrix X using kernel K.
def corr2d(X, K): 
	"""Compute 2D cross-correlation."""
	h, w = K.shape
	Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
	return Y

# class to perform convolution as a nn layer, store corresponding parameters appropriately.
class Conv2D(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.weight = nn.Parameter(torch.rand(kernel_size))
		self.bias = nn.Parameter(torch.zeros(1))
	def forward(self, x):
		return corr2d(x, self.weight) + self.bias

# now lets try to automatically find the gradient of an instance of the Conv2D class.
conv2d_instance = Conv2D(kernel_size = (3,3))
temp_tensor = torch.randn(size = (5,5), dtype = torch.float, requires_grad = True) # a tensor which we will pass through conv2d_instance
forward_pass_result = conv2d_instance(temp_tensor)
l = forward_pass_result.sum() # now, lets compute a loss and gradient on that
l.backward() # hmm so this occured successfully
conv2d_instance.weight.grad
conv2d_instance.bias.grad
# both these also ran successfully. Hence, I dont know what the error should have been. Looking 
# at the discussion a bit, it seems like it was meant to be something related to the fact that 
# we are performing in-place operations on the tensor Y in the function corr2d, and it is being 
# accessed later too. But I don't see how it is being accessed later and neither am I getting an
# error. Hence, leaving this for now.












