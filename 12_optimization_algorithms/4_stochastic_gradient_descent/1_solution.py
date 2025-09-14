# imports
import math
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# defining function and its gradient (using the ones mentioned in the chapter)
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2

# declaring the optimum of this function
optimum = (0,0)

# defining the stochastic gradient function (as done in the chapter)
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

# compute the euclidean distance between two tuples 
def Dist(t1, t2):
	return math.sqrt((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2)

# function to make the plot of distance from optimum with respect to num_iterations
def PlotDistFromOptimumVsIterations(results, optimum):
	# for each entry in results, we compute the distance of it from optimum and store in a list
	dists = []
	for i in range(len(results)):
		dists.append(Dist(results[i], optimum))
	plt.plot(dists)
	plt.show()

# trying out different learning rate schedules and plotting distance from optimum with 
# respect to num_iterations

# constant lr
def constant_lr():
    return 1
eta = 0.1
lr = constant_lr  # Constant learning rate
results = d2l.train_2d(sgd, steps=100, f_grad=f_grad)
d2l.show_trace_2d(f, results)
plt.show()
PlotDistFromOptimumVsIterations(results, optimum)

# exponential lr
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)
t = 1
lr = exponential_lr  
results = d2l.train_2d(sgd, steps=100, f_grad=f_grad)
d2l.show_trace_2d(f, results)
plt.show()
PlotDistFromOptimumVsIterations(results, optimum)

# polynomial lr
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)
t = 1
lr = polynomial_lr
results = d2l.train_2d(sgd, steps=100, f_grad=f_grad)
d2l.show_trace_2d(f, results)
plt.show()
PlotDistFromOptimumVsIterations(results, optimum)

# CONCLUSION: FOR THIS FUNCTION AND ITS GRADIENT (y = x1^2 + 2x2^2), THE POLYNOMIAL LEARNING 
# RATE SCHEDULER SEEMS TO WORK BETTER THAN CONSTANT LR AND EXPONENTIAL. WITH AROUND 50 EPOCHS, 
# IT IS THE ONLY ONE CLOSE TO THE OPTIMUM AND WITH 100 EPOCHS, ITS WITHIN (0.1,0.1) OF THE OPTIMAL.
# THE REASON IS AS STATED IN THE CHAPTER- EXPONENTIAL DECAYS THE LEARNING RATE TOO RAPIDLY, AND 
# CONSTANT LEADS TO OVERCORRECTING EVEN WHEN WE ARE CLOSE TO THE OPTIMUM. POLYNOMIAL SEEMS TO BE 
# A GOOD BALANCE BETWEEN BOTH.












