# imports
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math

def train_2d(trainer, steps=20, f_grad=None): 
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results): 
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                          torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

# version of train_2d for newton's method usage in the gradient function
def train_2d_newton(f_grad, f_hess, steps=20): 
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = gd_2d_newton(x1, x2, s1, s2, f_grad, f_hess)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

# version of gradient descent using newton's method with preconditioning (using diagonal 
# hessian instead of full hessian)
def gd_2d_newton(x1, x2, s1, s2, f_grad, f_hess):
    g1, g2 = f_grad(x1, x2)
    gradient_mat = np.array([[g1],[g2]])
    hessian_mat = f_hess(x1, x2)
    hessian_diag_inv = GetDiagonalInverseOfMatrix(hessian_mat)
    new_coordinate_2d_mat = eta * np.matmul(hessian_diag_inv, gradient_mat)
    # reshape the new_coordinate_2d_mat to get in appropriate form 
    return (new_coordinate_2d_mat[0][0], new_coordinate_2d_mat[1][0], 0, 0)

# return the hessian matrix given x1 and x2
def f_hess(x1, x2):
    return np.array([[2, 0],[0, 0.2]])

# take a square matrix, convert it into a diagonal matrix (make its non diagonal entries 0), and 
# make its inverse
def GetDiagonalInverseOfMatrix(x):
    # get diagonal version of a matrix
    x_diag = np.diag(np.diag(x))
    # now reciprocate diagonal entries
    bool_diag = np.diag(np.repeat(True, x_diag.shape[0])) # boolean diagonal matrix used as mask in next line
    x_diag_inv = np.reciprocal(x_diag, out = np.zeros(x_diag.shape), where = bool_diag)
    return x_diag_inv

### Now, using a function in which different coordinates vary at very different rates
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 0.1 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 0.2 * x2)

eta = 0.1
show_trace_2d(f_2d, train_2d_newton(f_2d_grad, f_hess))
plt.show()

## output:
# [(-5, -2),
#  (-0.5, -0.2),
#  (-0.05, -0.020000000000000004),
#  (-0.005000000000000001, -0.0020000000000000005),
#  (-0.0005000000000000001, -0.00020000000000000006),
#  (-5.0000000000000016e-05, -2.0000000000000012e-05),
#  (-5.000000000000002e-06, -2.000000000000001e-06),
#  (-5.000000000000002e-07, -2.0000000000000012e-07),
#  (-5.0000000000000024e-08, -2.0000000000000017e-08),
#  (-5.0000000000000026e-09, -2.0000000000000018e-09),
#  (-5.000000000000002e-10, -2.000000000000002e-10),
#  (-5.000000000000003e-11, -2.000000000000002e-11),
#  (-5.000000000000003e-12, -2.0000000000000024e-12),
#  (-5.000000000000003e-13, -2.0000000000000028e-13),
#  (-5.000000000000003e-14, -2.0000000000000028e-14),
#  (-5.000000000000004e-15, -2.000000000000003e-15),
#  (-5.000000000000004e-16, -2.000000000000003e-16),
#  (-5.000000000000005e-17, -2.000000000000003e-17),
#  (-5.000000000000005e-18, -2.000000000000003e-18),
#  (-5.000000000000005e-19, -2.0000000000000028e-19),
#  (-5.000000000000005e-20, -2.000000000000003e-20)]

### Conclusion: As one can see from the figure and output, just after 4 epochs, we are super 
# close to the optimal point (0,0). This is in contrast to the result from question 3, where 
# even after 20 epochs, we were far from the optimal point. This difference is due to 
# preconditioning.







