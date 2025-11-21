# imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

# the model and its params (here, being defined from scratch, because we want to play around 
# with all the intricacies of training on multiple gpus) 
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')

# set up all the functions and be able to train on multiple GPUs.

# just creating a version of d2l.load_data_fashion_mnist() here, so that its easy to refer to, 
# and I don't have to install d2l on modal instance
def load_data_fashion_mnist(batch_size):
	mnist_train = torchvision.datasets.FashionMNIST(root = 'my_vol/data', train = True, 
		transform = transforms.ToTensor(), download = False)
	mnist_test = torchvision.datasets.FashionMNIST(root = 'my_vol/data', train = False,
		transform = transforms.ToTensor(), download = False)
	# return a tuple of two dataloader objects, one for the mnist train data and another
	# for mnist test data
	return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle = True), # here, I am using num_workers = 0 (default), which means no extra subprocesses are used to load the data (and it is loaded in the main process).
			torch.utils.data.DataLoader(mnist_test, batch_size, shuffle = True))

# defining the d2l sgd function here for easy reference and not needing to install d2l.
# the nn sgd 
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# transfer the given iterable of tensors to provided device, make all its elements to require_grad
# and return the newly created iterable in the provided device.
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
    	# p = new_params[0] # example shape = torch.Size([20, 1, 3, 3])
        p.requires_grad_() # change p.requires_grad from False to True inplace
    return new_params

# Aggregate all the elements in iterable data (stored in different devices), and make all the 
# copies in the different devices to have that aggregated tensor. 
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

# split the provided tensors X and y, into the provided devices evenly. Uses nn.parallel.scatter.
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

# class For accumulating sums over `n` variables.
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    # add the supplied arguments elementwise to self.data
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy_gpu(net, data_iter, device, dev_params):
    """Compute the accuracy for a model on a dataset using a GPU. Defined in :numref:`sec_utils`"""

    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode

    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            # X, y = next(iter(data_iter))
            X = X.to(device)
            y = y.to(device)

            predictions = torch.argmax(net(X, dev_params), dim = 1)
            correct_predictions = torch.sum(y == predictions)
            metric.add(correct_predictions, y.numel()) # a side fact: y.numel() is a scalar and 
            # is computed from the tensor and immediately stored in CPU (we are not assigning it 
            # to a variable but just the command y.numel() causes it to be stored on cpu. This 
            # is not special about y.numel() but normally this is only what happens. We don't have 
            # to assign the results to something for it to be actually stored on some device)
    return metric[0] / metric[1]

# perform one step of training on a given batch on multiple GPUs. This is the version from the 
# book, which possibly has a flaw of not synchronizing the devices before calling allreduce()
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices) # X_shards and y_shards are tuples of tensors subsetted to the different devices
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
        	# i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
        	# i = 0
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
    	sgd(param, lr,  X.shape[0]) # Here, we use a full-size batch
    	# its probably a bit tricky to use the standard nn.optim.SGD here as that class's 
    	# instance needs to be told the parameters first, then loss.backward() called and then 
    	# .step() called. Hence, we would have to do something extra here to incorporate the 
    	# aggregation of gradients.

# the training function for training data from fashion_mnist on multiple gpus.
# Uses a fixed number of epochs defined in the function
def train(num_gpus, batch_size, lr, num_epochs = 5):
	# get dataloader objects for training data and test data
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [torch.device('cuda', i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            # X, y = next(iter(train_iter)) # X.shape = [256, 1, 28, 28], y.shape = [256]
            train_batch(X, y, device_params, devices, lr)
            # torch.cuda.synchronize() # this is optional only
        # print the accuracy of the model
        accuracy = evaluate_accuracy_gpu(net = lenet, data_iter = test_iter, 
            device = devices[0], dev_params = device_params[0])
        print(f'epoch: {epoch}, accuracy: {accuracy}')

# some global variables
batch_size = 256
lr = 0.01
num_gpus = 2

# call the train function and see if it runs
train(2, batch_size, lr)

# now, we start an instance with more gpus (say 4) on modal, and time how much time it takes 
# for 5 epochs when we train on 1,2,3,4 gpus. Actually the time it will take will likely be 
# the same, but it will reach higher accuracies faster. Each time, we change batch_size 
# appropriately too, because we need each gpu to be dealing with the same sized batch. We would 
# have ideally wanted to increase lr also with more gpus, as we are getting a better estimate 
# of the gradients when we are utilizing more gpus (due to a larger batch) for each step. 
# However, here we don't need to do that as that is already accounted for in the train_batch 
# function.

# HERE... TOMO START FROM INITIALIZING A MODAL CONFIG FILE FOR MORE GPUS (4 MAYBE), AND DO WHAT
# I HAVE WRITTEN ABOVE.

# 1 gpu
%time train(1, batch_size*1, lr) # max acc: 0.1, time: 31.4s

# 2 gpu
%time train(2, batch_size*2, lr) # max acc: 0.1, time: 32.9s

# 3 gpu
%time train(3, batch_size*3, lr) # max acc: 0.18, time: 33.2s

# 4 gpu
%time train(4, batch_size*4, lr) # max acc: 0.14, time: 32.5s

# hence, we can see that even with 5 epochs, the difference is clear and our hypothesis of 
# same time for more gpus per epoch, but high accuracy faster was also true. The difference 
# would have been more clear if we trained for more epochs.























