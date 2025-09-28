import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# defining function and its gradient. x is a 1d tensor here.
# x can be of any length, and accordingly, the function will be 1 upon powers of 2 
# multiplied by each entry of the vector.
def f(x):  # Objective function
    coefficients = 1/torch.pow(2, torch.arange(start = 0, end = len(x), step = 1))
    return 0.5*torch.sum(coefficients*x*x)

# version of same function as above, but it takes two numbers instead of one tensor as input 
# for the purpose of making the trace for it as the trace function expects it to take two inputs
def f_2d(x1, x2):
    return 0.5*(x1*x1 + 0.5*x2*x2)

# (A + A^T)x, which turns out to be Ax in this case, as A is diagonal and there is 1/2 at 
# start of the function
def f_grad(x):  # Gradient of the objective function
    # gradient is basically the matrix corresponding to the quadratic equation of the 
    # function we have above (which is a diagonal matrix with 1 upon powers of 2 on the diagonal)
    # multiplied by x
    coefficients = 1/torch.pow(2, torch.arange(start = 0, end = len(x), step = 1))
    A = torch.diag(coefficients)
    return torch.matmul(A, x)

# defining the stochastic gradient function. x is a 1d tensor here.
def sgd(x, f_grad, lr):
    g = f_grad(x) # g is also a 1d tensor
    # Simulate noisy gradient
    g += torch.normal(mean = torch.full(size = x.shape, fill_value = 0, dtype = torch.float), std = torch.full(size = x.shape, fill_value = 1, dtype = torch.float))
    # return the tensor with corrected values of x
    return x - lr*g

# function to take in the sgd function and do the training. 
# x is the input vector initialized to its initial value
def train_2d(trainer, x, lr, f_grad, steps=20):
    """Optimize a 2D objective function with a customized trainer."""
    results = [x]
    for i in range(steps):
        # i = 0
        x = trainer(x, f_grad, lr)
        results.append(x)
    print(f'epoch {i + 1}, x: {x}')
    return results

# perform the gradient descent using just stochastic gradient descent (mimicked in this 
# analytical function by adding normal noise to the determined gradient values at each step)
x = torch.tensor([100,100,100,100,100], dtype = torch.float) # starting with a 3d input
train_2d(sgd, x, 2, f_grad, steps = 40) # seems to diverge/not converge, atleast for some dimensions
train_2d(sgd, x, 1, f_grad, steps = 40) # better performance, but not exactly 0 reached, as the last dimension is slow to converge due to same learning rate
x = torch.tensor([100,100], dtype = torch.float)
d2l.show_trace_2d(f_2d, train_2d(sgd, x, 1, f_grad, steps = 20))
plt.show()

### WITH MOMENTUM

# momentum version of sgd
# prev_velocity is the velocity of the last step (it will be of same shape as x). 
# beta is the parameter for momentum. its between 0 and 1, and it being higher means we are 
# averaging gradients over more past steps
def sgd_momentum(x, f_grad, lr, prev_velocity, beta):
    g = f_grad(x) # g is also a 1d tensor
    # Simulate noisy gradient
    g += torch.normal(mean = torch.full(size = x.shape, fill_value = 0, dtype = torch.float), std = torch.full(size = x.shape, fill_value = 1, dtype = torch.float))
    # add to prev_velocity appropriately to get current velocity
    v = beta*prev_velocity + lr*g
    # return the tensor with corrected values of x
    return (x-lr*v, v)

# momentum version of train_2d function
# function to take in the sgd function and do the training. 
# x is the input vector initialized to its initial value
def train_2d_momentum(trainer, x, lr, beta, f_grad, steps=20):
    """Optimize a 2D objective function with a customized trainer."""
    results = [x]
    prev_velocity = torch.zeros(size = x.shape)
    for i in range(steps):
        # i = 0
        x, v = trainer(x, f_grad, lr, prev_velocity, beta)
        results.append(x)
        prev_velocity = v
    print(f'epoch {i + 1}, x: {x}')
    return results

# perform the gradient descent using just stochastic gradient descent (mimicked in this 
# analytical function by adding normal noise to the determined gradient values at each step)
x = torch.tensor([100,100,100,100,100], dtype = torch.float) # starting with a 3d input
train_2d_momentum(sgd_momentum, x, 2, 0.5, f_grad, steps = 40) # seems to diverge even more intensely than without momentum
train_2d_momentum(sgd_momentum, x, 1, 0.5, f_grad, steps = 40) # comes close to optimal solution (considering all dimensions) much faster than without momentum
x = torch.tensor([100,100], dtype = torch.float)
d2l.show_trace_2d(f_2d, train_2d_momentum(sgd_momentum, x, 1, 0.5, f_grad, steps = 20))
plt.show()

# CONCLUSION: WE CAN SEE FROM THE TRACES HOW THE MOMENTUM METHOD TAKES US TO THE OPTIMA IN A 
# MORE "CURVED" WAY AND HAS LESS OF SUDDEN CHANGES. HOWEVER, WITHOUT THE MOMENTUM, THE CHANGES 
# IN THE TRAJECTORY ARE MORE SUDDEN. DOING THE COMPARISON IN MORE DIMENSIONS MAKES THE DIFFERENCE
# IN PERFORMANCE BETWEEN WITH/WITHOUT MOMENTUM MORE CLEAR AS MOMENTUM APPROPRIATELY MODULATES 
# THE SPEED OF CONVERGENCE IN DIFFERENT DIMENSIONS WHEREAS NON-MOMENTUM, OUR LR IS SAME IN ALL 
# DIMENSIONS. PRECONDITIONING ALSO OFFERS THIS, BUT MOMENTUM ALSO HAS A DENOISING EFFECT IN SGD 
# (DUE TO AVERAGING OVER PREVIOUS STEPS, SO TAKING ADVANTAGE OF DATA FROM PREVIOUS STEPS TOO 
# IN SGD) WHICH PRECONDITIONING DOESN'T HAVE.






