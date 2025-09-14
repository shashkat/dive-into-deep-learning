# imports
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math

def show_trace_2d(f, results): 
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                          torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

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
def train_2d(trainer, steps=20, f_grad=None):  #@save
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
def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

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

def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 0.1 * x2 ** 2
def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 0.2 * x2)

### function, its grad and hessian values post rotation
def f_2d_rotated(x1, x2):  # Objective function
    return 1.5*x1*x1 + 1.5*x2*x2 + x1*x2
def f_2d_grad_rotated(x1, x2):  # Gradient of the objective function
    return (3*x1 + x2, 3*x2 + x1)
# return the hessian matrix given x1 and x2
def f_hess_rotated(x1, x2):
    return np.array([[3, 1],[1, 3]])

eta = 0.1
show_trace_2d(f_2d_rotated, train_2d_newton(f_2d_grad_rotated, f_hess_rotated))
plt.show()

# also using non preconditioned version
eta = 0.1
show_trace_2d(f_2d_rotated, train_2d(gd_2d, f_grad=f_2d_grad_rotated))
plt.show()

# Conclusion: Interestingly, by rotating the function along z axis by 45 degrees, we have 
# eliminated the difference in how it scales along the two coordinates (x and y axes) and made 
# it symmetric (in the sense of slope). This is why now, even without newtons method with 
# preconditioning, we are still able to reach the optimal point in around 20 epochs. Probably 
# with more finetuning of the eta value (single value for both coordinates), it will be reached 
# faster. Hence, this rotation strategy is a possible strategy of countering the imbalance in 
# scaling of the target value along two different coordinates.


