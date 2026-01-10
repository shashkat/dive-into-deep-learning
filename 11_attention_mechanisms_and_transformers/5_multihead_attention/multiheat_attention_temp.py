import torch
from torch import nn

# masked softmax func
def MaskedSoftMax(X, valid_lens):
	"""Perform softmax operation by masking elements on the last axis."""
	# X: 3D tensor, valid_lens: 1D or 2D tensor
	# shape of X: (num_batches, num_queries, num_keys)
	# if valid lens is 2d, it should have shape (num_batches, num_queries)
	# if valid lens is 1d, it should have the information for valid_lens for all queries corresponding to each batch, meaning shape (num_batches)
	def _sequence_mask(X, valid_len, value=0):
		maxlen = X.size(1) # this is num_keys, as X passed into this func is in 2d form (num_queries*num_batches, num_keys)
		mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
		X[~mask] = value
		return X
	if valid_lens is None:
		return nn.functional.softmax(X, dim=-1)
	else:
		shape = X.shape
		if valid_lens.dim() == 1:
			valid_lens = torch.repeat_interleave(valid_lens, shape[1])
		else:
			valid_lens = valid_lens.reshape(-1)
	# On the last axis, replace masked elements with a very large negative
	# value, whose exponentiation outputs 0
	X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
	return nn.functional.softmax(X.reshape(shape), dim=-1)

# scaled dot product attention class
class DotProductAttention(nn.Module): 
	"""Scaled dot product attention."""
	def __init__(self, dropout):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

	# Shape of queries: (batch_size, no. of queries, d)
	# Shape of keys: (batch_size, no. of key-value pairs, d)
	# Shape of values: (batch_size, no. of key-value pairs, value dimension)
	# Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
	def forward(self, queries, keys, values, valid_lens=None):
		d = queries.shape[-1]
		# Swap the last two dimensions of keys with keys.transpose(1, 2)
		scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
		self.attention_weights = masked_softmax(scores, valid_lens)
		return torch.bmm(self.dropout(self.attention_weights), values)

# create the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
	# define the init method
	def __init__(self, num_heads, dropout, num_hiddens, bias = False):
		super().__init__()
		self.num_heads = num_heads
		self.attention = DotProductAttention(dropout)
		self.W_q = nn.LazyLinear(out_features1 = num_hiddens, bias = bias)
		self.W_k = nn.LazyLinear(out_features = num_hiddens, bias = bias)
		self.W_v = nn.LazyLinear(out_features = num_hiddens, bias = bias)
		self.W_o = nn.LazyLinear(out_features = num_hiddens, bias = bias) # a bit strange that they are using num_hiddens are the output dimension too, but just keeping it consistent for now

	# now define the forward method
	def forward(self, queries, keys, values, valid_lens):


temp = torch.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])

temp.reshape(3,2,2)

temp.permute(0,2,1)

temp = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3)
temp.weight.shape















