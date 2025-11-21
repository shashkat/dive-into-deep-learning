# imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda import Stream

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
        # since the tensor already resides on gpu, there is no concept of pinning to memory
    return new_params

# Aggregate all the elements in iterable data (stored in different devices), and make all the 
# copies in the different devices to have that aggregated tensor. 
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

def allreduce_part1(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)

def allreduce_part2(data):
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

# does same thing as allreduce just aggregates in different gpus
def allreduceV2_part1(data, aggregation_device_index):
    # add to aggregation device's contents, the contents from all other devices
    for i in range(0, len(data)):
        if (i == aggregation_device_index): continue # if i currently points to aggregation_device's index, then do nothing
        data[aggregation_device_index][:] += data[i].to(data[aggregation_device_index].device)

# does same thing as allreduce just aggregates in different gpus
def allreduceV2_part2(data, aggregation_device_index):
    # copy from aggregation device, the contents to all other devices
    for i in range(0, len(data)):
        if (i == aggregation_device_index): continue # if i currently points to aggregation_device's index, then do nothing
        data[i][:] = data[aggregation_device_index].to(data[i].device)

# aggregating to one GPU, a certain parameter from all other GPUs. The senders here will work 
# on secondary streams, but the receiver will work on its default stream.
# secondary_streams is a list of streams on each device, which we use for the sending of data 
# from a particular GPU to the aggregation GPU.
def allreduceV3_part1(data, aggregation_device_index, secondary_streams):
    # add to aggregation device's contents, the contents from all other devices
    for c in range(0, len(data)):
        if (c == aggregation_device_index): continue # if c currently points to aggregation_device's index, then do nothing
        with torch.cuda.stream(secondary_streams[c]), torch.cuda.stream(secondary_streams[aggregation_device_index]): # the cth stream would be on device c, so effectively, the sender device is sending on a secondary stream, but the receiver is receiving on the default stream only, as the data is getting aggregated to the same piece of memory
            data[aggregation_device_index][:] += data[c].to(data[aggregation_device_index].device, non_blocking = True) # non_blocking argument has no effect when we transfer data from one GPU to another GPU, but only when CPU is involved. 

# distributing the aggregated data from aggregation GPU to all other GPUs
def allreduceV3_part2(data, aggregation_device_index, secondary_streams):
    # copy from aggregation device, the contents to all other devices
    for c in range(0, len(data)):
        if (c == aggregation_device_index): continue # if c currently points to aggregation_device's index, then do nothing
        with torch.cuda.stream(secondary_streams[c]), torch.cuda.stream(secondary_streams[aggregation_device_index]): # the cth stream would be on device c, so effectively, the receiver device is receiving on a secondary stream, but the sender is sending on the default stream only, as the same data is getting accessed in each send
            data[c][:] = data[aggregation_device_index].to(data[c].device) # non_blocking argument has no effect when we transfer data from one GPU to another GPU, but only when CPU is involved. 

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

# Compute the accuracy for a model on a dataset using a GPU
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
            allreduce_part1([device_params[c][i].grad for c in range(len(devices))])
        for i in range(len(device_params[0])):
            # i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
            # i = 0
            allreduce_part2([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        sgd(param, lr,  X.shape[0]) # Here, we use a full-size batch
        # its probably a bit tricky to use the standard nn.optim.SGD here as that class's 
        # instance needs to be told the parameters first, then loss.backward() called and then 
        # .step() called. Hence, we would have to do something extra here to incorporate the 
        # aggregation of gradients.

# v2 of train_batch. Here, we perform the aggregation of gradients on different gpus, and try 
# to split the workload evenly.
def train_batchV2(X, y, device_params, devices, lr):
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
            # here, train_batchV2 is different from train_batch. As i increases here, we perform 
            # the aggregation on a different device, and loop back once we have reached len(devices)
            allreduceV2_part1([device_params[c][i].grad for c in range(len(devices))], i%len(devices))
        for i in range(len(device_params[0])):
            # i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
            # i = 0
            # here, train_batchV2 is different from train_batch. As i increases here, we perform 
            # the aggregation on a different device, and loop back once we have reached len(devices)
            allreduceV2_part2([device_params[c][i].grad for c in range(len(devices))], i%len(devices))
    # The model parameters are updated separately on each GPU
    for param in device_params:
    	sgd(param, lr,  X.shape[0]) # Here, we use a full-size batch
    	# its probably a bit tricky to use the standard nn.optim.SGD here as that class's 
    	# instance needs to be told the parameters first, then loss.backward() called and then 
    	# .step() called. Hence, we would have to do something extra here to incorporate the 
    	# aggregation of gradients.

# v3 of train_batch. Difference from v2 is it calls allreduceV3 instead of allreduceV2 
# (performs the transfer in a parallelized fashion)
def train_batchV3(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices) # X_shards and y_shards are tuples of tensors subsetted to the different devices
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
        for X_shard, y_shard, device_W in zip(
            X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        # aggregate (add) the gradients of parameter i from all devices into device indexed i%num_devices
        for i in range(len(device_params[0])):
            # i = 0
            # i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
            allreduceV3_part1([device_params[c][i].grad for c in range(len(devices))], i%len(devices), secondary_streams_ixc[i])

        # synchronize all the secondary streams on senders, to make sure all aggregations have happened
        for c in range(len(devices)):
            torch.cuda.synchronize(c)

        # distribute aggregated gradients of parameter i from device indexed i%num_devices into all devices
        for i in range(len(device_params[0])):
            # i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
            # i = 0
            allreduceV3_part2([device_params[c][i].grad for c in range(len(devices))], i%len(devices), secondary_streams_ixc[i])
        
        # synchronize all the secondary streams on receivers, to make sure all distributions have happened
        for c in range(len(devices)):
            torch.cuda.synchronize(c)
    # The model parameters are updated separately on each GPU
    for param in device_params:
        sgd(param, lr,  X.shape[0]) # Here, we use a full-size batch
        # its probably a bit tricky to use the standard nn.optim.SGD here as that class's 
        # instance needs to be told the parameters first, then loss.backward() called and then 
        # .step() called. Hence, we would have to do something extra here to incorporate the 
        # aggregation of gradients.

# the training function for training data from fashion_mnist on multiple gpus.
# Uses a fixed number of epochs defined in the function. I supply the train_batch_fn here, 
# in order to test the different train_batch functions, which are different in how the gradients
# are aggregated (in same or different devices, and non_blocking behaviour of sender/receiver)
def train(num_gpus, batch_size, lr, num_epochs, train_batch_fn):
	# get dataloader objects for training data and test data
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [torch.device('cuda', i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            # X, y = next(iter(train_iter)) # X.shape = [256, 1, 28, 28], y.shape = [256]
            train_batch_fn(X, y, device_params, devices, lr)
            # torch.cuda.synchronize() # this is optional only

            #temp (was used when profiling to look where the time is being used)
            # with torch.profiler.profile(
            #     activities=[
            #         torch.profiler.ProfilerActivity.CPU, # profile the activity of cpu
            #         torch.profiler.ProfilerActivity.CUDA, # profile the activity of cuda (gpu)
            #     ],
            #     # specify the number of wait, warmup, active steps in the schedule.
            #     schedule=torch.profiler.schedule(
            #         wait=1, warmup=1, active=2, repeat=1, skip_first=1
            #     ),
            # ) as prof:
            #     for step_idx in range(1, 5):
            #         train_batch_fn(X, y, device_params, devices, lr)
            #         for c in range(len(devices)):
            #             torch.cuda.synchronize(c)
            #         prof.step()
            # prof.export_chrome_trace(f"trace_train_batchv3.json")

        # print the accuracy of the model
        accuracy = evaluate_accuracy_gpu(net = lenet, data_iter = test_iter, 
            device = devices[0], dev_params = device_params[0])
        print(f'epoch: {epoch}, accuracy: {accuracy}')

# some global variables
lr = 0.01
num_gpus = 4
batch_size = 256 * num_gpus

# create a list of lists of secondary streams on all devices
secondary_streams_ixc = [] # ixc denotes the shape of the list of lists. it has i (num_params) rows, and c (num_devices) cols.
for i in range(len(params)):
    secondary_streams_ixc.append([Stream(device = torch.device('cuda', c)) for c in range(num_gpus)])

# call the train function with different train_batch_fn and check the time taken

# simple train_batch (aggregation of gradients on 1 gpu)
%time train(4, 256*4, lr, 5, train_batch) # 33.7s, 33.4s

# train_batch with aggregation happening on different gpus, but all data transfers 
# blocking on sender and receiver
%time train(4, 256*4, lr, 5, train_batchV2) # 33.3s, 31.6s

# train_batch with aggregation happening on different gpus, with all data transfers non-blocking 
# on sender and receiver using streams.
%time train(4, 256*4, lr, 5, train_batchV3) # 38.7s, 38.7s

# Conclusion: So I played around a lot with how to transfer data between GPUs in a parallelized
# fashion, and train_batchV3 does that only. However, here the majority time is not taken during 
# the transfer and due to more streams, the overhead introduced in trying to parallelize transfers 
# is more than the gain. Hence, Here we see almost same times for when distributing the load of 
# transfer between GPUs without parallelization, but increase when adding parallelization due 
# to streams.






### old stuff

# HERE... SO MAKING THE AGGREGATION STEP WITH NEW STREAMS ON THE AGGREGATION GPU DID NOT INCREASE 
# SPEED, WHEREAS I EXPECTED IT WOULD INCREASE SPEED. IT COULD BE BECAUSE THE TRANSFER IS STILL 
# BLOCKING ON THE SENDER. I DIDN'T PIN THE MEMORY BEING TRANSFERRED HERE, BECAUSE IT WAS 
# ORIGINALLY RESIDING IN GPU, AND THERE IS NO CONCEPT OF PINNING IN THE GPU. HENCE, I MIGHT 
# BE MISSING SOMETHING ON THAT. A LESS LIKELY POSSIBILITY IS THAT THE AGGREGATION STEP WASNT 
# TAKING MUCH TIME TO BEGIN WITH AND THE REDISTRIBUTION STEP WAS TAKING MORE TIME. NEXT, I 
# WILL TEST HOW TO TRANSFER DATA FROM 1 GPU TO ANOTHER, IN A PARALLEL FASHION FOR BOTH OF THEM. 
# WILL NEED TO DO SOME PROFILING FOR THIS.

# now, we start an instance with more gpus (say 4) on modal, and time how much time it takes 
# for 5 epochs when we train on 1,2,3,4 gpus. Actually the time it will take will likely be 
# the same, but it will reach higher accuracies faster. Each time, we change batch_size 
# appropriately too, because we need each gpu to be dealing with the same sized batch. We would 
# have ideally wanted to increase lr also with more gpus, as we are getting a better estimate 
# of the gradients when we are utilizing more gpus (due to a larger batch) for each step. 
# However, here we don't need to do that as that is already accounted for in the train_batch 
# function.

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

# newer version of allreduce function that aggregates different parameters on different gpus. 
# But each data transfer is blocking for both sender and receiver.
# def allreduceV2(data, aggregation_device_index):
    # add to aggregation device's contents, the contents from all other devices
    # for i in range(0, len(data)):
    #     if (i == aggregation_device_index): continue # if i currently points to aggregation_device's index, then do nothing
    #     data[aggregation_device_index][:] += data[i].to(data[aggregation_device_index].device)
    
    # # copy from aggregation device, the contents to all other devices
    # for i in range(0, len(data)):
    #     if (i == aggregation_device_index): continue # if i currently points to aggregation_device's index, then do nothing
    #     data[i][:] = data[aggregation_device_index].to(data[i].device)

# aggregation step with no_blocking = True for sender, and when data is being aggregated on a 
# device, it uses a different stream on that device for each transfer
def allreduceV4_part1(data, aggregation_device_index):
    # add to aggregation device's contents, the contents from all other devices
    for i in range(0, len(data)):
        # create a new stream on the aggregation device 
        s = Stream(device = torch.device('cuda', aggregation_device_index))

        # if i currently points to aggregation_device's index, then do nothing
        if (i == aggregation_device_index): continue
        
        # do the transfer in the context of the stream
        with torch.cuda.stream(s):
            data[aggregation_device_index][:] += data[i].to(data[aggregation_device_index].device, non_blocking = True)

# distribution step with no_blocking = True for sender
def allreduceV4_part2(data, aggregation_device_index):
    # copy from aggregation device, the contents to all other devices
    for i in range(0, len(data)):
        if (i == aggregation_device_index): continue # if i currently points to aggregation_device's index, then do nothing
        data[i][:] = data[aggregation_device_index].to(data[i].device, non_blocking = True)


# v4 of train_batch. Difference from v3 is it ensures that the transfer is non-blocking in the 
# receiver too
def train_batchV4(X, y, device_params, devices, lr):
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
            # aggregate (add) the gradients of parameter i from all devices into device 
            # indexed i%num_devices
            allreduceV4_part1([device_params[c][i].grad for c in range(len(devices))], i%len(devices))
        torch.cuda.synchronize() # ensure that all the gradients have been aggregated in some or the other device
        for i in range(len(device_params[0])):
            # i corresponds to the index of a particular parameter across all devices, eg- bias of layer 2.
            # i = 0
            # distributed aggregated gradients of parameter i from device indexed i%num_devices 
            # into all devices
            allreduceV4_part2([device_params[c][i].grad for c in range(len(devices))], i%len(devices))
        torch.cuda.synchronize() # ensure that all the gradients have been distributed appropriately among the devices before actually correcting parameter values 
    # The model parameters are updated separately on each GPU
    for param in device_params:
        sgd(param, lr,  X.shape[0]) # Here, we use a full-size batch
        # its probably a bit tricky to use the standard nn.optim.SGD here as that class's 
        # instance needs to be told the parameters first, then loss.backward() called and then 
        # .step() called. Hence, we would have to do something extra here to incorporate the 
        # aggregation of gradients.



















