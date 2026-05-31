# This is a simple CNN which can be used for testing various types of attributes/training behaviours on some 
# training data like FashionMNIST

from torch import nn
import torch

# the lenet model architecture
class LeNet(nn.Module):
	# method to initialize an instance of this class, where we indicate what all attributes to 
	# store in the instance
	def __init__(self, input_channels, num_classes):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, padding = 2),
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size=2), # stride is by default equal to kernel_size
			nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5), # no padding here
			nn.Sigmoid(),
			nn.AvgPool2d(kernel_size = 2),
			# second block start, that makes the image into a flattened vector and works on it
			nn.Flatten(),
			nn.LazyLinear(out_features = 120),
			nn.Sigmoid(),
			nn.LazyLinear(out_features = 84),
			nn.Sigmoid(),
			nn.LazyLinear(out_features = num_classes)
		)

	# now, define the forward method, where we indicate how the forward pass is done
	def forward(self, X):
		return self.net(X)

# version with each module separately as an attribute of the instance. This helps in having names of all the modules 
# as we assign them, instead of something like net.0, which is what happens in the above model initialization
# Note that it has exact same architecture as above though
class LeNet(nn.Module):
	
	# method to initialize an instance of this class, where we indicate what all attributes to 
	# store in the instance
	def __init__(self, input_channels, num_classes):
		super().__init__()
		# shapes are given as comments here with assumed initial image shape as 28x28. But it can possibly work with other input shapes too.
		self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, padding = 2) # shape after this remains same: 28x28
		self.sig1 = nn.Sigmoid()
		self.avgpool1 = nn.AvgPool2d(kernel_size=2) # stride is by default equal to kernel_size. shape: 14x14
		self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5) # no padding here. shape: 10x10
		self.sig2 = nn.Sigmoid()
		self.avgpool2 = nn.AvgPool2d(kernel_size = 2) # shape: 5x5
		self.flatten1 = nn.Flatten()
		self.lin1 = nn.LazyLinear(out_features = 120)
		self.sig3 = nn.Sigmoid()
		self.lin2 = nn.LazyLinear(out_features = 84)
		self.sig4 = nn.Sigmoid()
		self.lin3 = nn.LazyLinear(out_features = num_classes)

	# now, define the forward method, where we indicate how the forward pass is done
	def forward(self, X):
		first_conv_result = self.avgpool1(self.sig1(self.conv1(X)))
		second_conv_result = self.avgpool2(self.sig2(self.conv2(first_conv_result)))
		flatten_result = self.flatten1(second_conv_result)
		linear1_result = self.sig3(self.lin1(flatten_result))
		linear2_result = self.sig4(self.lin2(linear1_result))
		return self.lin3(linear2_result)
	

