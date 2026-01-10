import torch

# some useful functions for performing convolution on image using kernel
def corr2d(X, K): 
	"""Compute 2D cross-correlation."""
	h, w = K.shape
	Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
	return Y

# create an image with diagonal edge
X = torch.tensor([
	[0,0,0,0,0],
	[0,0,0,0,1],
	[0,0,0,1,1],
	[0,0,1,1,1],
	[0,1,1,1,1],
	[1,1,1,1,1]
])

# declare the kernel
K = torch.tensor([[1, -1]])

# now, use the kernel on the image
corr2d(X, K)
# tensor([[ 0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0., -1.],
#         [ 0.,  0., -1.,  0.],
#         [ 0., -1.,  0.,  0.],
#         [-1.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.]])
# Hence, we see that the diagonal edge has been detected successfully. This means that this 
# "horizontal version" of the finite difference operator is able to detect both, vertical and 
# diagonal edges (basically all edges with a vertical component).

#####################################
############ TRANSPOSE X ############
#####################################
X2 = X.T
# tensor([[0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 1, 1, 1],
#         [0, 0, 1, 1, 1, 1],
#         [0, 1, 1, 1, 1, 1]])

corr2d(X2, K)
# tensor([[ 0.,  0.,  0.,  0., -1.],
#         [ 0.,  0.,  0., -1.,  0.],
#         [ 0.,  0., -1.,  0.,  0.],
#         [ 0., -1.,  0.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.]])
# Hence, even after transposing X, the diagonal edge has been detected successfully, which was 
# expected as even after transpose, the diagonal edge would still remain diagonal

#####################################
############ TRANSPOSE K ############
#####################################

K2 = K.T
# tensor([[ 1],
#         [-1]])

corr2d(X, K2)
# tensor([[ 0.,  0.,  0.,  0., -1.],
#         [ 0.,  0.,  0., -1.,  0.],
#         [ 0.,  0., -1.,  0.,  0.],
#         [ 0., -1.,  0.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.]])
# Again, the diagonal edge has been detected successfully, even with the transposed kernel. 
# This was also expected by me, as after transposing the kernel it becomes a horizontal edge 
# (or edge having a horizontal component) detector. Since a diagonal edge has a horizontal 
# component too, the kernel works in this case too.













