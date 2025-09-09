import torch
import d2l
from torch import dropout, nn
import math

# declare some example variables
batch_size = 100
num_queries = 5
num_keys_values = 10
query_dimension = 3
key_dimension = 4
value_dimension = 2
queries = torch.randn((batch_size, num_queries, query_dimension))
keys = torch.randn((batch_size, num_keys_values, key_dimension))
values = torch.randn((batch_size, num_keys_values, value_dimension))
valid_lens = torch.randint(low = 1, high = num_keys_values + 1, size = (batch_size, num_queries)) # any entry of valid_lens cannot be greater than num_keys

# masked softmax func
def MaskedSoftMax(X, valid_lens): #@save
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

# define the distance-based version of dotproductattention
class DotProductAttentionDistance(nn.Module):
	# define the __init__ method
	def __init__(self, dropout):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

	# now define the forward method
	# Shape of queries: (batch_size, no. of queries, d)
	# Shape of keys: (batch_size, no. of key-value pairs, d)
	# Shape of values: (batch_size, no. of key-value pairs, value dimension)
	# Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
	def forward(self, queries, keys, values, valid_lens = None):
		# first, we compute qkt - 1/2(||k||^2)
		raw_weights = torch.bmm(queries, torch.transpose(keys, 1, 2)) - torch.square(torch.linalg.vector_norm(keys, dim = -1).unsqueeze(dim = 1))/2 # shape = (batch, queries, keys)
		# now, we correct the mean to be 0 (currently -d/2) and variance to be 1 (currently 3d/2)
		d = queries.shape[2]
		scaled_weights = (raw_weights + d/2)/math.sqrt((3*d)/2)
		# now that we have the scaled weights, we compute masked softmax on it
		softmaxed_weights = MaskedSoftMax(scaled_weights, valid_lens)
		# finally, we have the softmax computed attention weights, and we can simply pass them through self.dropout, and multiply by values
		return torch.bmm(self.dropout(softmaxed_weights), values) # shape = (batch, queries, value_dimension)
temp = DotProductAttentionDistance(0.3)
temp(queries, keys, values, valid_lens)

class AdditiveAttention(nn.Module):
	# define the __init__ method
	def __init__(self, dropout, query_dimension, key_dimension): # not implementing in the lazy way. Maybe later on can explore that possibility. For now, non-lazy seems fine.
		super().__init__()
		self.dropout = nn.Dropout(p = dropout)
		# declare the linear layer which will be used to deal with queries and keys of different dimensions
		self.W = nn.Linear(in_features = query_dimension, out_features = key_dimension, bias = False)

	# define the forward method
	# Shape of queries: (batch_size, no. of queries, d)
	# Shape of keys: (batch_size, no. of key-value pairs, d)
	# Shape of values: (batch_size, no. of key-value pairs, value dimension)
	# Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
	def forward(self, queries, keys, values, valid_lens):
		# just multiply the queries and keys with the W appropriately to get the raw attention weights
		raw_weights = torch.bmm(self.W(queries), torch.transpose(keys, 1, 2))
	 	# not scaling to ensure 0 mean and 1 variance as it wasn't considered in the AdditiveAttention func in the book either
	  	# just compute the masked softmax
		softmaxed_weights = MaskedSoftMax(raw_weights, valid_lens)
	  	# now, pass the softmaxed_weights to the dropout layer, and multiply with values using torch bmm
		return torch.bmm(self.dropout(softmaxed_weights), values)

temp = AdditiveAttention(0.3, query_dimension, key_dimension)
temp(queries, keys, values, valid_lens).shape



temp = nn.Linear(in_features = query_dimension, out_features = key_dimension, bias = False)
torch.bmm(temp(queries), torch.transpose(keys, dim0 = 1, dim1 = 2)).shape



# THE DotProductAttentionDistance IS WORKING BUT JUST DOING NON DISTANCE THING. TOMO GO THROUGH 
# THE MASKEDDOTPLOT FUNC AND MAKE THE CHANGES TO DotProductAttentionDistance FUNC TO ACTUALLY 
# ANSWER THE QUESTION IN EXERCISE.

# testing reshape functionality
temp = torch.tensor([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]], dtype = torch.float)
torch.reshape(temp, (-1, temp.shape[-1]))
temp.reshape(-1, temp.shape[-1])

temp2 = torch.tensor([1,2,3], dtype = torch.float)
temp2.shape
temp2.repeat_interleave(4)

temp[0].norm(dim = 0)
torch.norm(temp[0], dim = 0)

temp2.norm()

torch.linalg.vector_norm(keys, dim = -1).unsqueeze(dim = 1)





