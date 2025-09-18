# Not using MXNet, but just modifying the current train_sgd function to divide lr by 10 post 
# each epoch
import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

# trainer function
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

# function to get the data
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1

# generic training function (with personal modification of stopping when loss < 0.25, and 
# also modification of dividing learning rate by 10 after each epoch)
def train_ch11(trainer_fn, states, hyperparams, data_iter,feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            
            # PERSONAL EDIT: if loss is less than 0.25, then simply return. We do this, because 
            # we want to be able to time when loss reaches a certain value and don't care about 
            # num_epochs or num_iters
            if l < 0.25:
            	print(f'reached loss < 0.25. Loss = {l}')
            	return

            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()

        # when we reach here, the epoch is over. 
        hyperparams['lr'] /= 10 # divide the lr by 10

    # print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

# train using stochastic gradient descent (by varying batch_size, can be made minibatch SGD 
# or vanilla GD)
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

# I will try a set of learning rate values and batch_size values in attempt to get fastest 
# (in terms of clock time instead of num_epochs or num_iterations) to a decent loss (<0.25)
learning_rates = [1, 0.1]
batch_sizes = [3, 10, 100, 400, 1500]
for learning_rate in learning_rates:
	# learning_rate = 0.1
	for batch_size in batch_sizes:
		starttime = time.time()
		train_sgd(learning_rate, batch_size, 100)
		endtime = time.time()
		print(f'--> lr = {learning_rate}, batch_size = {batch_size}, time taken = {round(endtime - starttime, 2)}')

# CONCLUSION- As we can see below, and compare with the previous solution, the time taken for 
# some configurations' (like lr = 0.1, batch_size = 10) learning rate and batch size has reduced 
# due to step reduction in LR (though for some, it has increased too):
# --> lr = 1, batch_size = 3, time taken = 22.62
# --> lr = 1, batch_size = 10, time taken = 0.03
# --> lr = 1, batch_size = 100, time taken = 0.04
# --> lr = 1, batch_size = 400, time taken = 0.09
# --> lr = 1, batch_size = 1500, time taken = 2.1
# --> lr = 0.1, batch_size = 3, time taken = 0.02
# --> lr = 0.1, batch_size = 10, time taken = 0.02
# --> lr = 0.1, batch_size = 100, time taken = 0.17
# --> lr = 0.1, batch_size = 400, time taken = 4.97
# --> lr = 0.1, batch_size = 1500, time taken = 2.1


