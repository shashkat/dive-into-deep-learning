# imports
import numpy as np
import matplotlib.pyplot as plt

# define the target function and sample points from it (with some noise added). y = 5x+10
def f(x):
	return (5*x + 10)

# we will parametrize our model as: y = mx + c. Hence, loss and loss_derivative funcs are as follows
def Loss(datapoint, params):
	y_pred = params[0]*datapoint[0] + params[1]
	return (y_pred - datapoint[1])**2

def LossDerivative(datapoint, params):
	m = params[0]
	c = params[1]
	x = datapoint[0]
	y = datapoint[1]
	derivative_wrt_m = 2*(m*x + c - y)*x
	derivative_wrt_c = 2*(m*x + c - y)
	# gradient clipping
	# if derivative_wrt_m > 100:
	# 	derivative_wrt_m = 100
	# if derivative_wrt_c > 100:
	# 	derivative_wrt_c = 100
	# if derivative_wrt_m < -100:
	# 	derivative_wrt_m = -100
	# if derivative_wrt_c < -100:
	# 	derivative_wrt_c = -100
	return (derivative_wrt_m, derivative_wrt_c)

# just a util func to correct the params given they and their derivative
def CorrectParams(params, loss_derivatives, eta):
	return (params[0] - eta*loss_derivatives[0], params[1] - eta*loss_derivatives[1])

# make a dataset, which has points sampled from the target distribution (and some small 
# noise is added to each point). By default samples points from normal (0,100) distribution
def GetDatapoints(n):
	data = []
	for _ in range(n):
		x = 10*np.random.randn()
		y = f(x)
		# add small noise to both x and y
		# x = x + np.random.randn()/3
		# y = y + np.random.randn()/3
		data.append((x, y))
	return data

# sample datapoints from given list, with or without replacement. number of datapoints sampled
# is same as the length of the list
def SampleDatapoints(data, replace):
	if replace:
		indices = np.random.randint(low = 0, high = len(data), size = len(data))
	else:
		indices = np.random.permutation(len(data))
	return [data[i] for i in indices]

# then, we define how we will take a step of optimization
def step(datapoint, params, eta):
	# params: (m, c) -> tuple
	# datapoint: (x, y) -> tuple

	# compute the loss for this point (difference between model prediction and actual value in 
	# datapoint) (this is for purpose of storing)
	loss = Loss(datapoint, params)

	# compute the derivative of loss wrt each param (tuple with dL/dm and dL/dc)
	loss_derivatives = LossDerivative(datapoint, params)

	# correct the two params accordingly
	params = CorrectParams(params, loss_derivatives, eta)
	return loss, params


# get plenty of points initially
data_full = GetDatapoints(2000)

# then, we actually do the fitting, once by sampling points with replacement, and then by 
# sampling points without replacement

# with replacement
data1 = SampleDatapoints(data_full, replace=True) # now subset with/without replacement to actually get the datapoints to loop through
params = (-10, -10) # init params
eta = 0.001 # learning rate
loss_values = []
for i, datapoint in enumerate(data1):
	loss, params = step(datapoint, params, eta)
	# save loss in a list for plotting
	loss_values.append(loss)
# plot
plt.plot(loss_values)
plt.show()
# final params = (5.009277087147903, 9.632888167161115)

# without replacement
data2 = SampleDatapoints(data_full, replace=False) # now subset with/without replacement to actually get the datapoints to loop through
params = (-10, -10) # init params
eta = 0.001 # learning rate
loss_values = []
for i, datapoint in enumerate(data2):
	loss, params = step(datapoint, params, eta)
	# save loss in a list for plotting
	loss_values.append(loss)
# plot
plt.plot(loss_values)
plt.show()
# final params = (5.016036071407265, 9.640639935009252)

# CONCLUSION: SO I NEED TO PUT MORE THOUGHT INTO WHAT KIND OF FUNCTION AND MODEL TO USE TO 
# TEST THE DIFFERENCE BETWEEN SAMPLING DATAPOINTS WITH/WITHOUT REPLACEMENT. I WOULD NEED A SETUP
# WHERE THERE ARE DIFFERENCES ONLY IN THE WAY THE DATA IS SAMPLED, AND EVERYTHING ELSE IS 
# WELL CONTROLLED. HERE, WHAT I AM OBSERVING IS THAT BOTH THE SAMPLING PROCEDURES PERFORM 
# EQUALLY WELL, AND THE SMALL DIFFERENCES ARE JUST DUE TO THE DIFFERENT DATA USED TO TRAIN 
# THE MODELS AND NOT THE SAMPLING PROCESS (THIS IS BECAUSE I SEE NO DIFFERENCE IN THE 
# PERFORMANCE OF THE TWO WHEN I DO THIS MULTIPLE TIMES). HOWEVER, I GET THE IDEA WHY THAT IS 
# THE CASE THAT WE LIKE TO SAMPLE POINTS WITHOUT REPLACEMENT. BY SAMPLING WITH REPLACEMENT, 
# WE MIGHT MISS OUT ON SOME POINTS AND WE MIGHT OVERTRAIN ON SOME. AND THERE IS SEEMINGLY NO 
# DISADVANTAGE OF SAMPLING POINTS WITHOUT REPLACEMENT.



