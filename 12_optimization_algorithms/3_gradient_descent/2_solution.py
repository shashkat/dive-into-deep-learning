# imports
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math

# function to show the trace of gradient descent
def show_trace(results, f):
	n = max(abs(min(results)), abs(max(results)))
	f_line = torch.arange(-n, n, 0.01)
	d2l.set_figsize()
	d2l.plot([f_line, results], [[f(x) for x in f_line], [
		f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x

# function to perform binary search with objective of minimizing func
def binary_search_minimize(low, high, func, tol=1e-2):
    """Finds the value x in [low, high] that minimizes func(x) using ternary/binary search.
    Assumes func is unimodal on [low, high].
    
    Args:
        low (float): Lower bound of search interval.
        high (float): Upper bound of search interval.
        func (callable): Function to minimize.
        tol (float): Desired absolute error tolerance on x.
    Returns:
        float: Value of x that minimizes func(x).
    """
    while high - low > tol:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        if func(mid1) < func(mid2):
            high = mid2
        else:
            low = mid1
    return (low + high) / 2

# find the right eta value by performing a binary search
# a is lower bound of search. b is upper bound of search
# x is the current point we are at.
def LearningRateBinarySearch(a, b, x, f, f_grad):
	# corner case when slope is already 0
	if f_grad(x) == 0: return 0

	# the minimum value of learning rate is always gonna be 0
	eta_min = 0

	# determine the upper bound of eta using a and b. If slope is negative, we use a to determine
	# else we use b to determine.
	if f_grad(x) > 0:
		eta_max = (x-a)/f_grad(x)
	else:
		eta_max = (x-b)/f_grad(x)

	# now, perform binary search in range (eta_min, eta_max) with purpose of minimizing value
	# of f(x - eta*f_grad(x))
	eta = binary_search_minimize(eta_min, eta_max, lambda eta: f(x - eta*f_grad(x)))
	return eta

# modification of vanilla gradient descent, where we determine the right eta at each step by  
# performing a binary search in a range of values to find the value which leads to most reduction
# in f(x) value.
# a and b are lower and upper bounds of where our minima lies. x_init should be between a and b
def gd_line_search(a, b, x_init, f, f_grad):
	x = x_init
	results = [x]
	for i in range(10):
		# i = 0
		# determine eta
		eta = LearningRateBinarySearch(a, b, x, f, f_grad)
		x -= eta * f_grad(x)
		results.append(float(x))
	print(f'epoch 10, x: {x:f}')
	return results

# standard gd func
def gd(x_init, eta, f_grad):
    x = x_init
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

# applying vanilla gradient descent and visualizing the trace
results = gd(20, 0.1, f_grad)
show_trace(results, f)
plt.show()

# applying modified gradient descent and visualizing the trace
results = gd_line_search(-100, 100, 20, f, f_grad)
show_trace(results, f)
plt.show()

### PART 1 ANSWER: For the binary search, we just need the derivative of the function at the 
# current value of x, if that is what the question means by needing derivatives. Once we know 
# the derivate (i.e. slope) at current value of x, we compute the value of f at some point in 
# both possible segments of the distance between a and b, and choose the one which yields a 
# smaller value.


### PART 2: TODO

### PART 3:

def f(x):  # Objective function
    return math.log(math.exp(x) + math.exp(-2*x-3))

def f_grad(x):  # Gradient (derivative) of the objective function
	term1 = 1/(math.exp(x) + math.exp(-2*x-3))
	term2 = math.exp(x) - 2*math.exp(-2*x-3)
	return term1*term2

# applying modified gradient descent and visualizing the trace
results = gd_line_search(-100, 100, -2, f, f_grad)
show_trace(results, f)
plt.show()












