# Not attempting currently. My prediction is that the difference between SGD and minibatch SGD 
# will be that in SGD, momentum offers more benefit due to the more noise, and as we incorporate 
# more and more datapoints in each batch, the counter-noisiness benefit that momentum has will 
# fade away, but each step will take lot more time too due to passing more data through the model 
# for each step. Hence, we will have to do a tradeoff to maximize the speed of convergence.