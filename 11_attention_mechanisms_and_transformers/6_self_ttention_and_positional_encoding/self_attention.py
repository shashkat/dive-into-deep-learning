import torch
from torch import nn

class PositionalEncoding(nn.Module): #@save
	"""Positional encoding."""
	def __init__(self, num_hiddens, dropout, max_len=1000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		# Create a long enough P
		self.P = torch.zeros((1, max_len, num_hiddens))
		X = torch.arange(max_len, dtype=torch.float32).reshape(
			-1, 1) / torch.pow(10000, torch.arange(
			0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
		self.P[:, :, 0::2] = torch.sin(X)
		self.P[:, :, 1::2] = torch.cos(X)

	def forward(self, X):
		X = X + self.P[:, :X.shape[1], :].to(X.device)
		return self.dropout(X)


torch.arange(0, 10, 2, dtype=torch.float32)
temp = torch.zeros((1, 1000, 10))
temp.shape

X = torch.arange(20, dtype=torch.float32).reshape(
	-1, 1) / torch.pow(10000, torch.arange(
	0, 2, 2, dtype=torch.float32) / 10)

X.shape

temp[:,:,0::2] = X
temp[:,:,0::2].shape

temp = torch.zeros((1000, 20, 10))

temp[:,:, ]

([None]*5)

# playing around with modifying tensors in place to see when we can compute .backward() and when can't

temp1 = torch.tensor([[1,2]], dtype = torch.float32, requires_grad = True)
temp2 = torch.tensor([[3],[4]], dtype = torch.float32, requires_grad = True)
temp1.shape
temp2.shape

temp3 = torch.matmul(temp1, temp2)
temp3[0][0] = 11.0
temp4 = temp3*3
temp4.backward()
temp1.grad
temp2.grad
temp3.grad

# seems like after modificatin of non-leaf tensor, the grads depending on it, i.e. above it in the 
# graph become zero, but its allowed. For leaf tensors, its not allowed only. 
# Need to confirm the dependingness with differen temp3 above.

temp1 = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float)
temp2 = torch.tensor([7,8], dtype = torch.float, requires_grad=True)
temp3 = torch.matmul(temp2, temp1)
temp3.backward(gradient=torch.ones(3))
temp2.grad

# playing around a bit with inplace modifications in tensor entries where the tensor is a part of 
# a computation graph
temp1 = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float)
temp2 = torch.tensor([7,8], dtype = torch.float, requires_grad=True)
temp3 = torch.matmul(temp2, temp1)
temp3.retain_grad()
temp3[2] = 69
# temp4[1] = 54
temp4 = temp3*temp3
temp4.retain_grad()
temp5 = temp4.sum()
temp5.backward()
temp4.grad
temp3.grad


x = torch.randn(5, requires_grad=True)
y = x.pow(2)
print(x.equal(y.grad_fn._saved_self))  # True
print(x is y.grad_fn._saved_self)  # True

y.grad_fn.__dir__()
y.grad_fn
id(y.grad_fn._saved_self)
id(x)

# investigating in-place operations

leaf = torch.tensor([1,2,3,4,5], dtype = torch.float, requires_grad = True)
middle = leaf*2 # creating an intermediate tensor because can't do at all, in-place operations on leaf tensors requiring grad

# change middle in-place
middle.multiply_(3)
# middle.add_(2)
middle[2] = 10

root = middle + 2 # the .grad_fn of root doesn't need to store leaf
root = middle.pow(2) # the .grad_fn of root will refer to leaf
root = middle.exp()
root = middle*2

root.sum().backward()

leaf.grad
root.grad_fn._saved_result
root.grad_fn

# Was understand this part in pytorch autograd mechanics: "Every in-place operation ... referenced by any other Tensor".
# try to understand that presence of above tensors or below tensors cause limitation in 
# in-place operations and actually try it out.

















