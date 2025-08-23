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

# get the accuracy of the model for the provided data's val_dataloader object
def Accuracy(model, data):
	acc = []
	for val_batch in data.val_dataloader():
		# val_batch = next(iter(data.val_dataloader()))
		Y_hat = model(*val_batch[:-1]) # shape (batch_size, num_classes)
		acc.append(model.accuracy(Y_hat, val_batch[-1]))
	return sum(acc)/len(acc)

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
            X = blk(X) # since we are not supplying any valid_lens here, we are not doing any masking. We are considering the attention between each key-query pair
        # at this point, shape of X: batch_size, num_patches+1 (including cls), num_hiddens
        return self.head(X[:, 0])

# The ViT class which encapsulates multiple VitBlocks
class ViTMean(d2l.Classifier):
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
            X = blk(X) # since we are not supplying any valid_lens here, we are not doing any masking. We are considering the attention between each key-query pair
        # at this point, shape of X: batch_size, num_patches+1 (including cls), num_hiddens
        return self.head(torch.mean(X[:,1:,:], dim = 1))

# training
img_size, patch_size = 48, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1

# initialize data
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
# we subset data to only 10% points using torch.data.utils.Subset
data.train = SubsetProportionData(data.train, 0.05) # datapoints from 60k to 3k
data.val = SubsetProportionData(data.val, 0.02) # fraction of datapoints out of 10k

### case when we derive the output from cls's embeddings
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
# train the model
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
# get accuracy
accuracy1 = Accuracy(model, data)

### case when we derive the output from avg of patches' embeddings
model = ViTMean(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
# train the model
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
# get accuracy
accuracy2 = Accuracy(model, data)

# Conclusion: The accuracy increased from 0.4080 in case of getting the output from cls embeddings
# to 0.4792 in case of getting the output from the average embedding of all the patches. I wonder 
# why that is the case. My guess is that in this case, since we are attending every patch by every 
# patch (because of no valid_lens supplied when passing X through blks in ViT), obtaining the final 
# output using average embeddings of all patches is more beneficial/informative than the cls 
# patch, which originates from random noise. However, when we do supply valid_lens (like in 
# sentence tranformers), then due to the fact that cls is in the first position and attends to 
# every other patch, it might be more informative to get the output from its embedding than the 
# average of patches' embeddings. Another possible reason could be that a bigger data or more 
# epoches may turn the tables here, as cls requires more "correction" because it is totally 
# initialized by random values.




























