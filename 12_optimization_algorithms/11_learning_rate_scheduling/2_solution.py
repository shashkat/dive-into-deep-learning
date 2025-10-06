import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l
import matplotlib.pyplot as plt

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        # epoch = 0
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            # i = 0
            # item = next(iter(train_iter))
            # X = item[0]
            # y = item[1]
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

net = net_fn()
lr = 0.1
num_epochs = 30
trainer = torch.optim.SGD(net.parameters(), lr)
scheduler = torch.optim.lr_scheduler.PolynomialLR(trainer, total_iters=num_epochs, power = 0.1) # using power = 0.1 instead of the 0.5 used in the chapter
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler)
plt.show()

### CONCLUSION:
# If we compare the result we got with what is in the book (which is for power = 0.5), we 
# observe that the performance in this specific case with power=0.1 is slightly worse, 
# especially in the later epochs, when the difference in learning rate is most between the two 
# cases. The conclusion is that there is no golden rule for the value of power, and generally 
# keeping it too slow would be good when we expect to reach the optimum faster, but keeping 
# it higher would be good if we expect more time to reach the optimum. I believe we like to 
# keep it on higher side as it is safer, because even if we reach the optimum quite fast, we 
# will eventually close in on it when the learning rate reduces a bit later. On the contrary 
# we might take a huge amount of time to reach the optimum (as our speed of approaching it 
# will also reduce) in case to very low power value.



### other stuff
# type(net)
# net.__dir__()
# net.cuda()
# net.ipu()
# net.half()
# temp = next(iter(train_iter))
# len(temp)
# temp[0].shape
# temp[1].shape
# train_iter.__dir__()
# train_iter.__len__()
# temp2 = next(iter(test_iter))
# test_iter.__dir__()
# test_iter.batch_size
# len(test_iter)
# len(train_iter)
# 235*256

# temp2[0].shape
# temp[1].__dir__()
# temp[1].values()
# import pandas as pd
# pd.Series(temp[1]).value_counts()
# 256-235
# loss.__dir__()
















