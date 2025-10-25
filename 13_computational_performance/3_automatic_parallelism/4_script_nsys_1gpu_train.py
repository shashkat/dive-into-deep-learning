# here, I will write the python script that I will profile using the nsys profile command. 
# This script will have the code to train the model in 2 GPUs

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

# get the fashion_mnist_dataset object
fmnist_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', 
	train = True,
	transform = transforms.ToTensor())

# get the dataloader corresponding to the dataset object we created above
fmnist_dataloader = DataLoader(dataset = fmnist_dataset, batch_size = 128, shuffle = True)

# now, we create a simple network which will try to understand the data
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10)
)

# store in device variable, which device will be our primary device
device = torch.device('cuda:0')

# move model to device
model = model.to(device)

# using nn dataparallel to make use of more than 1 gpu for training now. Though it is generally 
# better to use DDP (distributed data parallel) instead, I am using DataParallel here just for 
# the simplicity of using it, and here I can demonstrate usage of more than 1 gpus and data 
# transfer between them. Ref: https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# model = nn.DataParallel(model) # not doing this so that training is just on 1 gpu

# now, we write the training function
def training_fn(net, num_epochs, dataloader, loss_fn, optimizer):
	# set the model to training mode
	net.train()

	# loop through the number of epochs and go through the whole data 
	for epoch in range(num_epochs):
		# epoch = 0

		# loop through each batch for current epoch
		for itr, datapoint in enumerate(dataloader):
			# itr = 0
			# datapoint = next(iter(dataloader))
			
			# get the X and y
			X, y = datapoint
			# move the X and y to gpu
			X = X.to(device)
			y = y.to(device)

			# pass X through model 
			preds = model(X) # shape = ([128, 10])

			# compute loss between preds of shape (128,10) and y of shape (128)
			loss = loss_fn(input = preds, target = y)

			# clear the gradients in the optimizer
			optimizer.zero_grad()

			# call the backward method on loss
			loss.backward()

			# perform optimizer step
			optimizer.step()

		# at the end of the epoch, print the loss
		print(f'End of epoch {epoch}. Loss = {round(loss.item(), 3)}')

# declare the loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.8)

# call the training function
training_fn(model, 10, fmnist_dataloader, loss_fn, optimizer)










