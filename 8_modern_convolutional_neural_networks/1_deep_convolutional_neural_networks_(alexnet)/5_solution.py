import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
import inspect
import collections

import wandb
run = wandb.init(project="d2l_8p5_alexnet_light") # assigning this is important in terminal ipython as else an error appears as it tries to print whatever is returned wand.init() which is not printable in terminal ipython

####################
##### general functions
####################

astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)

def accuracy(Y_hat, Y, averaged=True):
	"""Compute the number of correct predictions.
	Defined in :numref:`sec_classification`"""
	Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
	preds = astype(argmax(Y_hat, axis=1), Y.dtype)
	compare = astype(preds == reshape(Y, -1), torch.float32)
	return reduce_mean(compare) if averaged else compare

def layer_summary(net, X_shape):
	X = torch.randn(*X_shape)
	for layer in net:
		X = layer(X)
		print(layer.__class__.__name__, 'output shape:\t', X.shape)

class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 1 channel only
def LoadDatasetFMNIST(train = True):
    train_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', train = train, 
        download = False, transform = transforms.ToTensor())
    return (train_dataset)

def init_cnn(module):
	"""Initialize weights for CNNs.

	Defined in :numref:`sec_lenet`"""
	if type(module) == nn.Linear or type(module) == nn.Conv2d:
		nn.init.xavier_uniform_(module.weight)

class DataModule(HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root, num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    # def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    #     """Defined in :numref:`sec_synthetic-regression-data`"""
    #     tensors = tuple(a[indices] for a in tensors)
    #     dataset = torch.utils.data.TensorDataset(*tensors)
    #     return torch.utils.data.DataLoader(dataset, self.batch_size,
    #                                        shuffle=train)

class FashionMNIST(DataModule):
	"""The Fashion-MNIST dataset.

	Defined in :numref:`sec_fashion_mnist`"""
	def __init__(self, root='/my_vol/data', batch_size=64, resize=(28, 28)):
		super().__init__(root=root)
		self.save_hyperparameters()
		trans = transforms.Compose([transforms.Resize(resize),
									transforms.ToTensor()])
		self.train = torchvision.datasets.FashionMNIST(
			root=self.root, train=True, transform=trans, download=False)
		self.val = torchvision.datasets.FashionMNIST(
			root=self.root, train=False, transform=trans, download=False)

	def text_labels(self, indices):
		"""Return text labels.
	
		Defined in :numref:`sec_fashion_mnist`"""
		labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
				  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
		return [labels[int(i)] for i in indices]

	def get_dataloader(self, train):
		"""Defined in :numref:`sec_fashion_mnist`"""
		data = self.train if train else self.val
		return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
										   num_workers=self.num_workers)

	# def visualize(self, batch, nrows=1, ncols=8, labels=[]):
	# 	"""Defined in :numref:`sec_fashion_mnist`"""
	# 	X, y = batch
	# 	if not labels:
	# 		labels = self.text_labels(y)
	# 	d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)

class Trainer(HyperParameters):
	"""The base class for training models with data.

	Defined in :numref:`subsec_oo-design-models`"""

	def __init__(self, max_epochs, num_gpus, gradient_clip_val=0):
		"""Defined in :numref:`sec_use_gpu`"""
		self.save_hyperparameters()
		# self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
		self.gpus = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
	
	def prepare_batch(self, batch):
		"""Defined in :numref:`sec_use_gpu`"""
		if self.gpus:
			batch = [a.to(self.gpus[0]) for a in batch]
		return batch

	def prepare_model(self, model):
		"""Defined in :numref:`sec_use_gpu`"""
		model.trainer = self
		# model.board.xlim = [0, self.max_epochs]
		if self.gpus:
			model.to(self.gpus[0])
		self.model = model

	def clip_gradients(self, grad_clip_val, model):
		"""Defined in :numref:`sec_rnn-scratch`"""
		params = [p for p in model.parameters() if p.requires_grad]
		norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
		if norm > grad_clip_val:
			for param in params:
				param.grad[:] *= grad_clip_val / norm

	def prepare_data(self, data):
		self.train_dataloader = data.train_dataloader()
		self.val_dataloader = data.val_dataloader()
		self.num_train_batches = len(self.train_dataloader)
		self.num_val_batches = (len(self.val_dataloader)
								if self.val_dataloader is not None else 0)

	def fit(self, model, data):
		self.prepare_data(data)
		self.prepare_model(model)
		self.optim = model.configure_optimizers()
		self.epoch = 0
		self.train_batch_idx = 0
		self.val_batch_idx = 0
		for itr, self.epoch in enumerate(range(self.max_epochs)):
			self.fit_epoch()
			print(f'--> epoch {itr} done!')

	def fit_epoch(self):
		"""Defined in :numref:`sec_linear_scratch`"""
		self.model.train()
		for batch in self.train_dataloader:
			# batch = next(iter(self.train_dataloader))
			loss = self.model.training_step(self.prepare_batch(batch))
			self.optim.zero_grad()
			with torch.no_grad():
				loss.backward()
				if self.gradient_clip_val > 0:  # To be discussed later
					self.clip_gradients(self.gradient_clip_val, self.model)
				self.optim.step()
			self.train_batch_idx += 1
		if self.val_dataloader is None:
			return
		self.model.eval()
		for batch in self.val_dataloader:
			# batch = next(iter(self.val_dataloader))
			with torch.no_grad():
				self.model.validation_step(self.prepare_batch(batch))
			self.val_batch_idx += 1

####################
##### chapter specific functions
####################

class AlexNet_simple(nn.Module, HyperParameters):
	def __init__(self, lr=0.1, num_classes=10):
		super().__init__()
		self.save_hyperparameters()
		self.net = nn.Sequential(
			nn.LazyConv2d(6, kernel_size=5, stride=1, padding=2), nn.ReLU(), 
			# nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LazyConv2d(36, kernel_size=5, padding=2), nn.ReLU(),
			# nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LazyConv2d(120, kernel_size=5, padding=2), nn.ReLU(),
			# nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
			nn.LazyConv2d(96, kernel_size=5, padding=2), nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(), # 13x13
			nn.LazyLinear(2048), nn.ReLU(), nn.Dropout(p=0.5),
			nn.LazyLinear(num_classes))
		self.net.apply(init_cnn)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, X):
		return self.net(X)

	def training_step(self, batch):
		Y_hat = self(*batch[:-1])
		Y = batch[-1]
		l = self.loss(Y_hat, Y)
		# self.plot('loss', l, train=True)
		wandb.log({'training_loss': l})
		print(f'loss = {l}')
		acc = accuracy(Y_hat, Y)
		wandb.log({'training_acc': acc})
		return l

	def validation_step(self, batch):
		Y_hat = self(*batch[:-1])
		Y = batch[-1]
		l = self.loss(Y_hat, Y)
		wandb.log({'validation_loss': l})
		acc = accuracy(Y_hat, Y)
		wandb.log({'validation_acc': acc})

	def configure_optimizers(self):
		"""Defined in :numref:`sec_classification`"""
		return torch.optim.SGD(self.parameters(), lr=self.lr)

####################
##### main code
####################

data = FashionMNIST(batch_size=128, resize=(28, 28))
model = AlexNet_simple(lr=0.01)
trainer = Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)

# CONCLUSION:
# HENCE, WITH SIMPLY KEEPING THE IMAGE SIZE TO 28X28, AND MODIFYING THE NUMBER OF OUTPUT CHANNELS 
# FOR EACH 2D CONVOLUTION LAYER ACCORDINGLY, WE WERE ABLE TO MAKE ALEXNET LIGHTER AND BETTER 
# HANDLE SMALLER IMAGES.







