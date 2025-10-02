import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math

# defining function and its gradient.
def f(x1, x2):  # Objective function
    return 0.1*x1*x1 + 2*x2*x2

# gradient of the function
def f_grad(x1, x2):
    return (0.2*x1, 4*x2)

# defining the stochastic gradient function. 
def sgd(x1, x2, f_grad, lr, s1_prev, s2_prev):
    eps = 1e-6
    g1, g2 = f_grad(x1, x2) 
    # Simulate noisy gradient
    g1 += torch.normal(mean = 0.0, std = 1.0, size = (1,)).item()
    g2 += torch.normal(mean = 0.0, std = 1.0, size = (1,)).item()
    # get the squared norm of gradient and increment s_prev by it
    s1 = s1_prev + g1*g1
    s2 = s2_prev + g2*g2
    # using s, compute effective lr for both directions
    lr_effective_1 = lr/math.sqrt(s1 + eps)
    lr_effective_2 = lr/math.sqrt(s2 + eps)
    # return the tensor with corrected values of x
    return (x1 - lr_effective_1*g1, x2 - lr_effective_2*g2, s1, s2)

# function to take in the sgd function and do the training. 
# x is the input vector initialized to its initial value
def train_2d(trainer, x1, x2, lr, f_grad, steps=20):
    """Optimize a 2D objective function with a customized trainer."""
    results = [(x1, x2)]
    s1 = 0
    s2 = 0
    for i in range(steps):
        # i = 0
        x1, x2, s1, s2 = trainer(x1, x2, f_grad, lr, s1, s2)
        s_prev = s
        results.append((x1, x2))
    print(f'epoch {i + 1}, x: {x}')
    return results

x1 = 10
x2 = 10
train_2d(sgd, x1, x2, 2, f_grad, steps = 40) # we have to keep the lr a bit higher (2) and then we reach close to 0,0. And we can see that the rate of convergence goes super down super fast, demonstrating the strict lr correcting nature of adagrad
d2l.show_trace_2d(f, train_2d(sgd, x1, x2, 2, f_grad, steps = 40))
plt.show()

### NOW, WITH THE ROTATED VERSION OF THE FUNCTION (AND ITS GRADIENT)
# multiplied by each entry of the vector.
def f_rotated(x1, x2):  # Objective function
    return 0.1*(x1+x2)*(x1+x2) + 2*(x1-x2)*(x1-x2)

# gradient of the function
def f_grad_rotated(x1, x2):
    dfdx1 = 0.2*(x1+x2) + 4*(x1-x2)
    dfdx2 = 0.2*(x1+x2) - 4*(x1-x2)
    return (dfdx1, dfdx2)

x1 = 10
x2 = 10
train_2d(sgd, x1, x2, 2, f_grad_rotated, steps = 40) # we basically see the same rate of convergence as the non-rotated case.
d2l.show_trace_2d(f_rotated, train_2d(sgd, x1, x2, 2, f_grad_rotated, steps = 40)) 
plt.show()

### CONCLUSION: AS THE ROTATION DIDN'T CREATE A DIFFERENCE IN PERFORMANCE, THIS INDICATES THAT 
# THE PERFORMANCE WAS ALREADY INDEPENDENT OF DIFFERENT RATES OF CHANGE ALONG THE TWO DIRECTIONS, 
# WHICH SHOWS THE POWER OF ADAGRAD IN COUNTERING THE ISSUES WHEN RATE OF CHANGE IN DIFFERENT 
# DIRECTIONS ARE DIFFERENT.


