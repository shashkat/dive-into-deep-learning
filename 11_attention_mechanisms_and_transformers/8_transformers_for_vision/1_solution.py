# imports
import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# get specified proportion of datapoints from data, which is of type torchvision.datasets.x.x, 
# eg: torchvision.datasets.mnist.FashionMNIST. proportion is between 0 and 1.
def SubsetProportionData(data, proportion):
	subset_size = int(len(data) * proportion)
	subset_indices = torch.randperm(len(data))[:subset_size].tolist()
	return torch.utils.data.Subset(data, subset_indices)

# implementing PatchEmbedding class 
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

# implementing VitMLP class (vision transformer version of MLP that was in transformer). Difference
# is that it uses GeLU instead of ReLU, and has dropout
class ViTMLP(nn.Module):
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

# ViTBlock class which encapsulates one unit of passing input (patch) through vision transformer (encoder)
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))

# The ViT class which encapsulates multiple VitBlocks
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

# training
patch_size = 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1

# loop through multiple img_sizes and record the time it takes in training them. Keeping min 
# img_size as 16, because keeping kernel_size fixed at 16 
time_taken = {}
for img_size in tqdm([16,32,64,96,128]):
	# img_size = 128
	
	# initialize data
	data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
	# we subset data to only 10% points using torch.data.utils.Subset
	data.train = SubsetProportionData(data.train, 0.02) # datapoints from 60k to 1200
	data.val = SubsetProportionData(data.val, 0.1) # datapoints from 10k to 1k
	
	# declare model and trainer 
	model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
	            num_blks, emb_dropout, blk_dropout, lr)
	trainer = d2l.Trainer(max_epochs=1)

	starttime = time.perf_counter()
	trainer.fit(model, data)
	endtime = time.perf_counter()

	time_elapsed = endtime - starttime
	time_taken[img_size] = time_elapsed

# make the plot
img_sizes = list(time_taken.keys())
training_times = list(time_taken.values())
plt.figure(figsize=(6, 4))
plt.plot(img_sizes, training_times, marker='o')
plt.xlabel('Image Size')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs. Image Size')
plt.grid(True)
plt.show()
plt.clf()

# Conclusion: The training time seems to increase polynomially with img_size. My guess is that 
# the relation is quadratic, due to the img_size increasing the size of each datapoint by its 
# square as it is the size of two dimensions of the tensor.



