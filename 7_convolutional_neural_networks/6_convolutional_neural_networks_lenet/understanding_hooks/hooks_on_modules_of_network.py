# ref- https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging

import torch
from torch import nn

# print entries from dir function of an object, which have a particular substring
def DirSubsetToSubstring(obj, substr):
	entries_with_substr = []
	for entry in dir(obj):
		if substr in entry:
			entries_with_substr.append(entry)
	return entries_with_substr

# create the LeNet model class
class LeNet(nn.Module):
	# method to initialize an instance of this class, where we indicate what all attributes to 
	# store in the instance
	def __init__(self, input_channels, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, padding = 2)
		self.sig1 = nn.Sigmoid()
		self.avgpool1 = nn.AvgPool2d(kernel_size=2) # stride is by default equal to kernel_size
		self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5) # no padding here
		self.sig2 = nn.Sigmoid()
		self.avgpool2 = nn.AvgPool2d(kernel_size = 2)
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

model_instance = LeNet(1, 10)

for name, param in model_instance.named_parameters():
	print(name)
	try:
		print(param.shape)
	except:
		print('no param')



model_instance.get_parameter()
temp = model_instance.named_parameters()
temp1 = next(temp)
temp2 = next(temp)
temp3 = next(temp)
temp4 = next(temp)
temp5 = next(temp)
temp2[0], temp2[1].shape

DirSubsetToSubstring(model_instance, 'param')



class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)
   
  
  def forward(self, x):
    x = self.relu(self.conv(x))
    x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated 
                                                                  #conv outside less than 0
      
    # print whether there is any negative grad
    x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any())))  
    return self.fc1(self.flatten(x))  

net = myNet()

for name, param in net.named_parameters():
	print(name)
	try:
		print(param.shape)
	except:
		print('no param')


# now, 
