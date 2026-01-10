- Yes, we can initialize all the weight parameters in linear and softmax regression to the same values. 

Case1:
Suppose we talk about a linear regression with single output. Here, we already have one output to begin with, and there is no hidden layer, where a 'collapse' of nodes can possibly take place. Hence, its fine to initialize all weights with same value. 

Case2:
Next, suppose we talk about a linear regression with multiple outputs. Here, we have different true values for all the outputs, unlike in a neural network, where we generally combine all the outputs into one value, and compute the loss of that with a single, true value that we have for that datapoint. Hence, we can basically imagine the linear regression with multiple outputs as many different single layer neural networks with one output. According to the explanation from Case1, it is fine here too, to initialize all the weights with the same value.

Case3: 
Now coming to softmax regression. Here, also, we don't have a single output, but multiple outputs, and hence the network can be viewed as many different neural networks, like the one in Case1, where each of them can have weights initialized with same values. And since they can be thought of as different networks, all the weights across all the networks can be initialized with the same value, and they will still learn.

Hence, it can be boiled down to the fact that these models/networks don't have a hidden layer, outputs from which are combined into a single value to compute the final loss, which enables us to be able to initialize their weights with the same value.