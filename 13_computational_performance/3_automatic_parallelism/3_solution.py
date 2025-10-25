# so here, I will try to train a simple model using 2 gpus and cpu. There will be communication
# between the devices in each iteration because the gradients computed by the split batch in 
# both the gpus need to be aggregated at the end of each iteration, which will happen in the cpu.

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
model = nn.DataParallel(model)

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

# In [106]: training_fn(model, 10, fmnist_dataloader,
#         ⋮ loss_fn, optimizer)
# End of epoch 0. Loss = 0.674
# End of epoch 1. Loss = 0.648
# End of epoch 2. Loss = 0.467
# End of epoch 3. Loss = 0.434
# End of epoch 4. Loss = 0.318
# End of epoch 5. Loss = 0.454
# End of epoch 6. Loss = 0.331
# End of epoch 7. Loss = 0.282
# End of epoch 8. Loss = 0.281
# End of epoch 9. Loss = 0.285

# Comparing the time taken for a single epoch when we use 1 GPU vs when we use 2 GPUs

################ 2 GPUs ###############
%time training_fn(model, 1, fmnist_dataloader, loss_fn, optimizer)
# CPU times: user 11.8 s, sys: 810 ms, total: 12.6 s
# Wall time: 9.68 s

################ 1 GPU ################
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
# declare the loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.8)
# train
%time training_fn(model, 1, fmnist_dataloader, loss_fn, optimizer)
# CPU times: user 5.8 s, sys: 150 ms, total: 5.95 s
# Wall time: 6.04 s

# CONCLUSION 1: So apparently, in this case, training on a single GPU is faster than training on 2 GPUs. Lets 
# try to increase batch size and see the effects

# get the dataloader corresponding to the dataset object we created above
fmnist_dataloader_2048 = DataLoader(dataset = fmnist_dataset, batch_size = 2048, shuffle = True)

################ 1 GPU ################
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
# declare the loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.8)
# train
%time training_fn(model, 1, fmnist_dataloader_2048, loss_fn, optimizer)
# CPU times: user 5.03 s, sys: 0 ns, total: 5.03 s
# Wall time: 5.32 s

################ 2 GPU ################
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
# use 2 gpus
model = nn.DataParallel(model)
# declare the loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.8)
# train
%time training_fn(model, 1, fmnist_dataloader_2048, loss_fn, optimizer)
# CPU times: user 5.32 s, sys: 100 ms, total: 5.42 s
# Wall time: 5.37 s

# FINAL CONCLUSION: AS WE DO A BIGGER BATCH SIZE, THE TIMES TAKEN BY TRAINING ON A SINGLE GPU 
# AND DOUBLE COME CLOSER. WITH AN EVEN BIGGER BATCH SIZE (AND POSSIBLY MORE GPUS), THE 
# DIFFERENCE WOULD BECOME BETTER IN FAVOR OF USING MORE GPUS. ALSO KEEP IN MIND THAT HERE, WE ARE
# SIMPLY USING DATAPARALLEL FROM TORCH, WHICH IS NOT THE BEST WAY OF TRAINING ON MULTIPLE GPUS. 
# IT PERFORMS MULTITHREADING ON MUTLTIPLE GPUS, WHEREAS DDP (DISTRIBUTED DATA PARALLEL) ACTUALLY 
# LAUNCHES DIFFERENT KERNELS ON DIFFERENT GPUS AND THAT REDUCES OVERHEAD OF AGGREGATING THE 
# RESULTS AT EACH STEP. BUT STILL THIS SCRIPT IS A NICE PRACTICE OF TRAINING ON MULTIPLE GPUS.




