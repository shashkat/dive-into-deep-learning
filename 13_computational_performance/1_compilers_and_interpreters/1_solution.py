# imports
from d2l import torch as d2l
from torch import nn
import torch
from tqdm import tqdm

# load the fashion mnist dataset
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size = batch_size)

net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 10, 13, 13)
	nn.Conv2d(in_channels=10, out_channels=5, kernel_size=2), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 5, 6, 6)
	nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 1, 2, 2),
	nn.Flatten(), # (batch_size, 4)
	nn.Linear(in_features=4, out_features=10) # (batch_size, 10)
	)
loss = nn.CrossEntropyLoss()
lr = 0.1
trainer = torch.optim.SGD(params = net.parameters(), lr = lr)
for epoch in tqdm(range(5)):
	for i, (X, y) in enumerate(train_iter):
		net.train()
		y_preds = net(X)
		l = loss(y_preds, y)
		trainer.zero_grad()
		l.backward()
		# print(f'loss = {l}')
		trainer.step()
# without torchscript: 31 secs, 32 secs

net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 10, 13, 13)
	nn.Conv2d(in_channels=10, out_channels=5, kernel_size=2), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 5, 6, 6)
	nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3), nn.ReLU(),
	nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 1, 2, 2),
	nn.Flatten(), # (batch_size, 4)
	nn.Linear(in_features=4, out_features=10) # (batch_size, 10)
	)
net = torch.jit.script(net)
loss = nn.CrossEntropyLoss()
lr = 0.1
trainer = torch.optim.SGD(params = net.parameters(), lr = lr)
for epoch in tqdm(range(5)):
	for i, (X, y) in enumerate(train_iter):
		net.train()
		y_preds = net(X)
		l = loss(y_preds, y)
		trainer.zero_grad()
		l.backward()
		# print(f'loss = {l}')
		trainer.step()
# with torchscript: 31 secs, 31 secs

# Probably with a much bigger network, the difference will be more clear



















