import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math

# rotated function
def f_rotated(x1, x2):  # Objective function
    return 0.1*(x1+x2)*(x1+x2) + 2*(x1-x2)*(x1-x2)

# gradient of the function
def f_grad_rotated(x1, x2):
    dfdx1 = 0.2*(x1+x2) + 4*(x1-x2)
    dfdx2 = 0.2*(x1+x2) - 4*(x1-x2)
    return (dfdx1, dfdx2)

# defining the stochastic gradient function. 
def sgd(x1, x2, f_grad, lr, gamma, s1_prev, s2_prev):
    eps = 1e-6
    g1, g2 = f_grad(x1, x2) 
    # Simulate noisy gradient
    g1 += torch.normal(mean = 0.0, std = 1.0, size = (1,)).item()
    g2 += torch.normal(mean = 0.0, std = 1.0, size = (1,)).item()
    # get the squared norm of gradient and increment s_prev, after appropriately weighing using gamma
    s1 = gamma*s1_prev + (1-gamma)*g1*g1
    s2 = gamma*s2_prev + (1-gamma)*g2*g2
    # using s, compute effective lr for both directions
    lr_effective_1 = lr/math.sqrt(s1 + eps)
    lr_effective_2 = lr/math.sqrt(s2 + eps)
    # return the tensor with corrected values of x
    return (x1 - lr_effective_1*g1, x2 - lr_effective_2*g2, s1, s2)

# function to take in the sgd function and do the training. 
# x is the input vector initialized to its initial value.
# gamma is the parameter indicating leaky averaging of s values in rmsprop.
def train_2d(trainer, x1, x2, lr, gamma, f_grad, steps=20):
    """Optimize a 2D objective function with a customized trainer."""
    results = [(x1, x2)]
    s1 = 0
    s2 = 0
    for i in range(steps):
        # i = 0
        x1, x2, s1, s2 = trainer(x1, x2, f_grad, lr, gamma, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x: {x}')
    return results

x1 = 10
x2 = 10
train_2d(sgd, x1, x2, 0.5, 0.9, f_grad_rotated, steps = 40) # first of all, it is converging. Secondly, the convergence is not slowing up rapidly (which would have happened in case of adagrad and would require us to use a much higher lr), which indicates pretty good performance.
d2l.show_trace_2d(f_rotated, train_2d(sgd, x1, x2, 0.5, 0.9, f_grad_rotated, steps = 20)) 
plt.show()





