from torch import nn
import torch
from torch.optim import optimizer
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# create the LeNet model class
class LeNet(nn.Module):
	# method to initialize an instance of this class, where we indicate what all attributes to 
	# store in the instance
	def __init__(self, input_channels, num_classes):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, padding = 2),
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size=2), # stride is by default equal to kernel_size
			nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5), # no padding here
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size = 2),
			# second block start, that makes the image into a flattened vector and works on it
			nn.Flatten(),
			nn.LazyLinear(out_features = 120),
			nn.Sigmoid(),
			nn.LazyLinear(out_features = 84),
			nn.Sigmoid(),
			nn.LazyLinear(out_features = num_classes)
		)

	# now, define the forward method, where we indicate how the forward pass is done
	def forward(self, X):
		return self.net(X)

# create the LeNet model class (change avgpool to maxpool and sigmoid to relu)
class LeNetV2(nn.Module):
	# method to initialize an instance of this class, where we indicate what all attributes to 
	# store in the instance
	def __init__(self, input_channels, num_classes):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2), # stride is by default equal to kernel_size
			nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5), # no padding here
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2),
			# second block start, that makes the image into a flattened vector and works on it
			nn.Flatten(),
			nn.LazyLinear(out_features = 120),
			nn.ReLU(),
			nn.LazyLinear(out_features = 84),
			nn.ReLU(),
			nn.LazyLinear(out_features = num_classes)
		)

	# now, define the forward method, where we indicate how the forward pass is done
	def forward(self, X):
		return self.net(X)

# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 
# 1 channel only
def LoadDatasetFMNIST(train = True):
    train_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', 
    	train = train, download = False, transform = transforms.ToTensor())
    return (train_dataset)

# train function
def train(model_instance, loss_fn, optimizer_instance, num_epochs):
	# now we go through the whole data num_epochs times
	for epoch in range(num_epochs):
		model_instance.train()
		train_losses = []
		for X, y in fmnist_dataloader:
			X = X.to('cuda')
			y = y.to('cuda')
			y_pred = model_instance(X)
			l = loss_fn(y_pred, y)
			train_losses.append(l)
			optimizer_instance.zero_grad()
			l.backward()
			optimizer_instance.step()

		# also check the loss on the validation data now
		val_losses = []
		with torch.no_grad():
			model_instance.eval()
			for X, y in fmnist_val_dataloader:
				X = X.to('cuda')
				y = y.to('cuda')
				y_pred = model_instance(X)
				l = loss_fn(y_pred, y)
				val_losses.append(l)
		
		# print at the end of epoch, the average value of loss
		avg_train_loss = float(sum(train_losses)/len(fmnist_dataloader))
		avg_val_loss = float(sum(val_losses)/len(fmnist_val_dataloader))
		print(f'Epoch {epoch}: train_l: {round(avg_train_loss, 3)}, val_l: {round(avg_val_loss, 3)}')

# get the dataset object for the fmnist data
fmnist_dataset = LoadDatasetFMNIST()
fmnist_val_dataset = LoadDatasetFMNIST(train = False)
# create a dataloader object from the dataset object as that will be actually used in the train 
# function in the loop
fmnist_dataloader = DataLoader(fmnist_dataset, batch_size = 128, shuffle = True)
fmnist_val_dataloader = DataLoader(fmnist_val_dataset, batch_size = 128, shuffle = True)

########## training the base LeNet
model_instance = LeNet(1, 10)
model_instance.to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer_instance = torch.optim.Adam(model_instance.parameters(), lr = 0.01)
# call the train function
train(model_instance, loss_fn, optimizer_instance, 10)
# In [139]: train(model_instance, loss_fn, optimizer_instance, 10)
# Epoch 0: train_l: 0.95, val_l: 0.56
# Epoch 1: train_l: 0.472, val_l: 0.46
# Epoch 2: train_l: 0.402, val_l: 0.421
# Epoch 3: train_l: 0.359, val_l: 0.394
# Epoch 4: train_l: 0.338, val_l: 0.369
# Epoch 5: train_l: 0.315, val_l: 0.351
# Epoch 6: train_l: 0.299, val_l: 0.339
# Epoch 7: train_l: 0.288, val_l: 0.34
# Epoch 8: train_l: 0.28, val_l: 0.314
# Epoch 9: train_l: 0.267, val_l: 0.317

########## training the LeNetV2 (with average pooling replaced by maxpooling and sigmoid 
# replaced by relu)
model_instance = LeNetV2(1, 10)
model_instance.to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer_instance = torch.optim.Adam(model_instance.parameters(), lr = 0.01)
# call the train function
train(model_instance, loss_fn, optimizer_instance, 10)
# In [47]: train(model_instance, loss_fn, optimizer_instance, 10)
# Epoch 0: train_l: 0.535, val_l: 0.405
# Epoch 1: train_l: 0.354, val_l: 0.364
# Epoch 2: train_l: 0.32, val_l: 0.326
# Epoch 3: train_l: 0.301, val_l: 0.34
# Epoch 4: train_l: 0.284, val_l: 0.323
# Epoch 5: train_l: 0.276, val_l: 0.345
# Epoch 6: train_l: 0.267, val_l: 0.321
# Epoch 7: train_l: 0.266, val_l: 0.323
# Epoch 8: train_l: 0.262, val_l: 0.303
# Epoch 9: train_l: 0.262, val_l: 0.339

# CONCLUSION: THE DIFFERENCE IS CLEAR. WITH RELU AND MAXPOOL REPLACING SIGMOID AND AVGPOOL 
# RESPECTIVELY, THE MODEL'S TRAINING LOSS DECREASES MUCH FASTER AND GOES MORE DOWN AT THE 
# END OF 10 EPOCHS. THE VALIDATION LOSS ALSO SEEMS TO FOLLOW A SIMILAR PATTERN.











