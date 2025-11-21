# this question is slightly vague. Hence, I take this opportunity to test why they didn't call 
# torch.cuda.synchronize() before calling allreduce() in train_batch function. 

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
def train(num_gpus, batch_size, lr):
	# get dataloader objects for training data and test data
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            # X, y = next(iter(train_iter)) # X.shape = [256, 1, 28, 28], y.shape = [256]
            train_batch(X, y, device_params, devices, lr)
            # torch.cuda.synchronize()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')

# some global variables
batch_size = 256
lr = 0.01

#######################################################
# testing if in the worst case, the train_batch function can infact throw an error, when the 
# two gpus haven't completed the backward step, but we want to aggregate the gradients.

# variation of train_batch, to allow me to do the tests
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices) # X_shards and y_shards are tuples of tensors subsetted to the different devices
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]

    # here, I give device 0 a big task, so that it is late in getting to the l.backward(), and 
    # possibly leads to a Nonetype error when we call allreduce below
    torch.mm(torch.randn(size = (10000, 10000), device = devices[0]), 
    	torch.randn(size = (10000, 10000), device = devices[0]))

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

# SO EVEN AFTER GIVING CUDA:0 A LOT OF TASK BEFORE GIVING BOTH GPUS THE LOSS.BACKWARD() 
# TASK, THERE IS NO ERROR THAT I WAS EXPECTING TO BE THERE WHEN WE CALL ALLREDUCE. SO SOMETHING 
# IS CAUSING SYNCHRONIZATION BETWEEN THE GPUS HAPPENING. MY GUESS IS THAT EVEN IF ONE THING IS 
# NONE, TORCH SEES THAT AS THE THING BEING ABSENT, AND THAT LEADS TO IT WAITING FOR IT (MEANING 
# SYNCHRONIZATION). TOMORROW, I CAN READ A BIT ABOUT WHEN SYNCHRONIZATION HAPPENS AUTOMATICALLY, 
# AND IF THAT INCLUDES NONETYPE ALSO. IF I FIND NOTHING OF THE SORT, I CAN CREATE AN ISSUE.

# CONCLUSION: SO WHEN WE CALL ALLREDUCE, INSIDE THE FUNCTION, WE ARE TRYING TO ACCESS THE 
# ELEMENTS OF THE SUPPLIED LIST IN THE DIFFERENT DEVICES (AND AGGREGATING THEM TO THE FIRST 
# ELEMENT). THIS OPERATION IS AUTOMATICALLY SYNCHRONIZING (FOR BOTH DEVICES INVOLVED, THE ONE 
# FROM WHICH DATA IS BEING ACCESSED AND THE ONE ONTO WHICH WE ARE AGGREGATING IT). THIS IS 
# BECAUSE TO BE ABLE TO ACCESS THE DATA, WE OBVIOUSLY NEED THE DEVICE TO BE DONE WITH ITS TASKS, 
# AND HENCE IDLE, AND TO AGGREGATE, WE NEED THE DEVICE (INDEXED 0) TO BE IDLE, AS THE AGGREGATION 
# IS GONNA HAPPEN ON IT ONLY. HENCE, EVEN WITHOUT ANY SPECIFIC SYNCHRONIZING COMMANDS, ALLREDUCE
# ALWAYS WORKS BECAUSE ITS OPERATIONS REQUIRE SYNCHRONIZATION. ALSO, I THOUGHT ABOUT IT AND MOST 
# LIKELY, THE TORCH.CUDA.SYNCHRONIZE() COMMAND AFTER CALLING THE TRAIN_BATCH FUNCTION IS ALSO 
# OPTIONAL, AS THE COMMANDS THAT WILL FOLLOW IN THE NEXT ITERATION WILL AUTOMATICALLY SYNCHRONIZE 
# THE DEVICES TOO, BUT IT IS NICE TO ENSURE THE SYNCHRONIZATION AFTER EACH EPOCH OURSELVES (AND 
# ALSO CAN BE HELPFUL FOR TIMING EACH EPOCH INDEPENDENTLY).

#######################################################
# testing what would happen if in allreduce function, the data argument had one element as None.
# Result: It will throw and error: AttributeError: 'NoneType' object has no attribute 'to'
temp1 = torch.randn(size = (5,4), device = torch.device('cuda:0'))
temp2 = None
data = [temp1, temp2]
allreduce(data)











