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
run = wandb.init(project="d2l_8p1_alexnet") # assigning this is important in terminal ipython as else an error appears as it tries to print whatever is returned wand.init() which is not printable in terminal ipython

####################
##### unused general functions
####################

class ProgressBoard(HyperParameters):
	"""The board that plots data points in animation.

	Defined in :numref:`sec_oo-design`"""
	def __init__(self, xlabel=None, ylabel=None, xlim=None,
				 ylim=None, xscale='linear', yscale='linear',
				 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
				 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
		self.save_hyperparameters()
	
	def draw(self, x, y, label, every_n=1):
		"""Defined in :numref:`sec_utils`"""
		Point = collections.namedtuple('Point', ['x', 'y'])
		if not hasattr(self, 'raw_points'):
			self.raw_points = collections.OrderedDict()
			self.data = collections.OrderedDict()
		if label not in self.raw_points:
			self.raw_points[label] = []
			self.data[label] = []
		points = self.raw_points[label]
		line = self.data[label]
		points.append(Point(x, y))
		if len(points) != every_n:
			return
		mean = lambda x: sum(x) / len(x)
		line.append(Point(mean([p.x for p in points]),
						  mean([p.y for p in points])))
		points.clear()
		if not self.display:
			return
		d2l.use_svg_display()
		if self.fig is None:
			self.fig = d2l.plt.figure(figsize=self.figsize)
		plt_lines, labels = [], []
		for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
			plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
										  linestyle=ls, color=color)[0])
			labels.append(k)
		axes = self.axes if self.axes else d2l.plt.gca()
		if self.xlim: axes.set_xlim(self.xlim)
		if self.ylim: axes.set_ylim(self.ylim)
		if not self.xlabel: self.xlabel = self.x
		axes.set_xlabel(self.xlabel)
		axes.set_ylabel(self.ylabel)
		axes.set_xscale(self.xscale)
		axes.set_yscale(self.yscale)
		axes.legend(plt_lines, labels)
		display.display(self.fig)
		display.clear_output(wait=True)

class Module(d2l.nn_Module, HyperParameters):
	"""The base class of models.

	Defined in :numref:`sec_oo-design`"""
	def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
		super().__init__()
		self.save_hyperparameters()
		self.board = ProgressBoard()

	def loss(self, y_hat, y):
		raise NotImplementedError

	def forward(self, X):
		assert hasattr(self, 'net'), 'Neural network is defined'
		return self.net(X)

	def plot(self, key, value, train):
		"""Plot a point in animation."""
		assert hasattr(self, 'trainer'), 'Trainer is not inited'
		self.board.xlabel = 'epoch'
		if train:
			x = self.trainer.train_batch_idx / \
				self.trainer.num_train_batches
			n = self.trainer.num_train_batches / \
				self.plot_train_per_epoch
		else:
			x = self.trainer.epoch + 1
			n = self.trainer.num_val_batches / \
				self.plot_valid_per_epoch
		self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
						('train_' if train else 'val_') + key,
						every_n=int(n))

	def training_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		self.plot('loss', l, train=True)
		return l

	def validation_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		self.plot('loss', l, train=False)

	def configure_optimizers(self):
		raise NotImplementedError

	def configure_optimizers(self):
		"""Defined in :numref:`sec_classification`"""
		return torch.optim.SGD(self.parameters(), lr=self.lr)

	def apply_init(self, inputs, init=None):
		"""Defined in :numref:`sec_lazy_init`"""
		self.forward(*inputs)
		if init is not None:
			self.net.apply(init)

class Classifier(nn.Module):
	"""The base class of classification models.

	Defined in :numref:`sec_classification`"""
	def validation_step(self, batch):
		Y_hat = self(*batch[:-1])
		self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
		self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

	def accuracy(self, Y_hat, Y, averaged=True):
		"""Compute the number of correct predictions.
	
		Defined in :numref:`sec_classification`"""
		Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
		preds = astype(argmax(Y_hat, axis=1), Y.dtype)
		compare = astype(preds == reshape(Y, -1), float32)
		return reduce_mean(compare) if averaged else compare

	def loss(self, Y_hat, Y, averaged=True):
		"""Defined in :numref:`sec_softmax_concise`"""
		Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
		Y = reshape(Y, (-1,))
		return F.cross_entropy(
			Y_hat, Y, reduction='mean' if averaged else 'none')

	def layer_summary(self, X_shape):
		"""Defined in :numref:`sec_lenet`"""
		X = randn(*X_shape)
		for layer in self.net:
			X = layer(X)
			print(layer.__class__.__name__, 'output shape:\t', X.shape)

# astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
# argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
# reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
# reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
# randn = torch.randn
# float32 = torch.float32

####################
##### general functions
####################

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

class AlexNet(nn.Module, HyperParameters):
	def __init__(self, lr=0.1, num_classes=10):
		super().__init__()
		self.save_hyperparameters()
		self.net = nn.Sequential(
			nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
			nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
			nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
			nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
			nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
			# nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5), # commented this out for faster training of alexnet
			nn.LazyLinear(num_classes))
		self.net.apply(init_cnn)
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, X):
		return self.net(X)

	def training_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		# self.plot('loss', l, train=True)
		wandb.log({'training_loss': l})
		print(f'loss = {l}')
		return l

	def validation_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		wandb.log({'validation_loss': l})
		# self.plot('loss', l, train=False)

	def configure_optimizers(self):
		"""Defined in :numref:`sec_classification`"""
		return torch.optim.SGD(self.parameters(), lr=self.lr)

####################
##### main code
####################

model = AlexNet(lr=0.01)
data = FashionMNIST(batch_size=128, resize=(224, 224))
trainer = Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)







