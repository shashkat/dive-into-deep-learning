# imports
import torch
from torch import nn
from d2l import torch as d2l

# implementing PatchEmbedding class on own for change
class PatchEmbedding(nn.Module):
	def __init__(self, img_size, patch_size, num_hiddens):
		super().__init__()

		# if x is not tuple or list, make into tuple, else just return x
		def _make_tuple(x):
			if not isinstance(x, (list, tuple)):
				return (x,x)
			return x

		# get the tuples of img_size and patch_size
		img_size = _make_tuple(img_size)
		patch_size = _make_tuple(patch_size)

		# compute the number of patches and add as attribute to object instance
		self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

		# the convolution layer, which is how we do a linear transformation of each patch of the 
		# image into a num_hiddens sized vector
		self.conv = nn.LazyConv2d(out_channels = num_hiddens, kernel_size = patch_size, stride = patch_size)


	def forward(self, X):
		# in_shape: num_batches, in_channels, input_height, input_weight
		return(self.conv(X).flatten(start_dim = 2).transpose(1,2)) # out shape: num_batches, num_patches, num_hiddens


patch_embeddor = PatchEmbedding(96, 16, 120)
patch_embeddor.num_patches

num_batches = 10
in_channels = 3
input_height = 96
input_weight = 96
X = torch.randn(num_batches, in_channels, input_height, input_weight)
out = patch_embeddor(X)
out.shape

# implementing VitMLP class (vision transformer version of MLP that was in transformer). Difference
# is that it uses GeLU instead of ReLU, and has dropout
class VitMLP(nn.Module):
	def __init__(self, vitmlp_num_hidden, num_hiddens, dropout = 0.5):
		super().__init__()
		self.linear1 = nn.LazyLinear(out_features = vitmlp_num_hidden)
		self.gelu = nn.GELU()
		self.dropout1 = nn.Dropout(p = dropout)
		self.linear2 = nn.LazyLinear(out_features = num_hiddens)
		self.dropout2 = nn.Dropout(p = dropout)

	def forward(self, X):
		# X shape = num_batches, num_patches, num_hiddens
		return self.dropout2(self.linear2(self.dropout1(self.gelu(self.linear1(X)))))

['hello']*3

x = torch.tensor([[1], [2], [3]])
x.size()
torch.Size([3, 1])
x.expand(6, 4)
temp = torch.randn(2,3,4,5)
temp[0].shape

d2l.



