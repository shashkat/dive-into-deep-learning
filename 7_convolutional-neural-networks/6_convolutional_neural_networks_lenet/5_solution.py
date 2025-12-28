# here, I take the trained model, and try to feed into it, a random noise image and see how the 
# activations look like. Also will try feeding into it, an image of something not related to the 
# dataset's (fashionmnist) images 

from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# create the LeNet model class. Having all modules separately as attributes of model makes it 
# easy to access them (useful here as we want to see activations post conv1 and conv2)
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

# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 
# 1 channel only
def LoadDatasetFMNIST(train = True):
    train_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', 
    	train = train, download = False, transform = transforms.ToTensor())
    return (train_dataset)

# function to load and return dataset (cifar10). Images have height and width 32x32, and have 
# 3 channels
def LoadDatasetCIFAR10(train = True):
    train_dataset = torchvision.datasets.CIFAR10(root = '/my_vol/data', 
    	train = train, download = False, transform = transforms.ToTensor())
    return (train_dataset)

# train function
def train(model_instance, loss_fn, optimizer_instance, num_epochs):
	# now we go through the whole data num_epochs times
	for epoch in range(num_epochs):
		model_instance.train()
		train_losses = []
		for X, y in fmnist_dataloader:
			X = X.to('cuda')
			y = y.to('cuda')
			y_pred = model_instance(X)
			l = loss_fn(y_pred, y)
			train_losses.append(l)
			optimizer_instance.zero_grad()
			l.backward()
			optimizer_instance.step()

		# also check the loss on the validation data now
		val_losses = []
		with torch.no_grad():
			model_instance.eval()
			for X, y in fmnist_val_dataloader:
				X = X.to('cuda')
				y = y.to('cuda')
				y_pred = model_instance(X)
				l = loss_fn(y_pred, y)
				val_losses.append(l)
		
		# print at the end of epoch, the average value of loss
		avg_train_loss = float(sum(train_losses)/len(fmnist_dataloader))
		avg_val_loss = float(sum(val_losses)/len(fmnist_val_dataloader))
		print(f'Epoch {epoch}: train_l: {round(avg_train_loss, 3)}, val_l: {round(avg_val_loss, 3)}')

# get the dataset object for the fmnist data
fmnist_dataset = LoadDatasetFMNIST()
fmnist_val_dataset = LoadDatasetFMNIST(train = False)
cifar10_dataset = LoadDatasetCIFAR10()
# create a dataloader object from the dataset object as that will be actually used in the train 
# function in the loop
fmnist_dataloader = DataLoader(fmnist_dataset, batch_size = 128, shuffle = True)
fmnist_val_dataloader = DataLoader(fmnist_val_dataset, batch_size = 128, shuffle = True)
cifar10_dataloader = DataLoader(cifar10_dataset, batch_size = 10, shuffle = True)

########## training the base LeNet
model_instance = LeNet(1, 10)
model_instance.to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer_instance = torch.optim.Adam(model_instance.parameters(), lr = 0.01)
# call the train function. We train the model instance before visualizing its activations for some example inputs as we want to see the results of the trained model rather than random convolution's results
train(model_instance, loss_fn, optimizer_instance, 10)

# now, lets see the activations of the model's first and second layers for different inputs
# for this, we will use forward hooks. So first we need the hook function
activations = {} # make a dict, in which the hook function will store the activations as forward propagation is done on the module that gets registered with our hook function
def hook_fn(m, i, o):
	activations[m] = o
# now to the model instance's conv1 and conv2, register our hook_fn
conv1_removable_handle = model_instance.conv1.register_forward_hook(hook_fn)
conv2_removable_handle = model_instance.conv2.register_forward_hook(hook_fn)

# now, lets see the activations for some example inputs through the model
temp_X, _ = next(iter(cifar10_dataloader))
temp_X = temp_X.to('cuda')
temp_X = temp_X[:, [0], :28, :28] # need to just choose one channel as the model is able to only deal with one channel. Also keep to just 28x28 pixels
model_instance(temp_X)

# lets see the actual images first
img_grid_actual_image = torchvision.utils.make_grid(temp_X, normalize = False, nrow = 5)
img_grid_actual_image = img_grid_actual_image.permute(1,2,0) # make channel dimension last as matplotlib likes it that way
img_grid_actual_image = img_grid_actual_image.to('cpu')
plt.figure(figsize=(8,4))
plt.title('Actual Images')
plt.imshow(img_grid_actual_image) # only seeing the particular channel of all images
plt.axis('off')
plt.savefig(f'/my_vol/cnn_activations_layer1_actual_images.png')
plt.clf()

# now, lets see the activations
temp = list(activations.values())[0]
img_grid = torchvision.utils.make_grid(temp[:10,:,:,:], normalize = True, nrow = 5)
img_grid = img_grid.permute(1,2,0) # make channel dimension last as matplotlib likes it that way
img_grid = img_grid.to('cpu')

for ch_num in range(img_grid.shape[2]):
	plt.figure(figsize=(8,4))
	plt.title('Activations in LeNet Convolution Layer 1')
	plt.imshow(img_grid[:, :, ch_num]) # only seeing the particular channel of all images
	plt.axis('off')
	plt.savefig(f'/my_vol/cnn_activations_layer1_channel{ch_num}.png')
	plt.clf()

# now, lets see the activations of layer 2
temp = list(activations.values())[1]
img_grid = torchvision.utils.make_grid(temp[:10,:,:,:], normalize = True, nrow = 5)
img_grid = img_grid.permute(1,2,0) # make channel dimension last as matplotlib likes it that way
img_grid = img_grid.to('cpu')

for ch_num in range(img_grid.shape[2]):
	plt.figure(figsize=(8,4))
	plt.title('Activations in LeNet Convolution Layer 1')
	plt.imshow(img_grid[:, :, ch_num]) # only seeing the particular channel of all images
	plt.axis('off')
	plt.savefig(f'/my_vol/cnn_activations_layer2_channel{ch_num}.png')
	plt.clf()

# CONCLUSIONS: With the images from cifar10, and especially for the layer2 output, it definitely 
# seems that the model is unable to keep intact, the patterns from the objects in the images. 
# This is expected, as likely the parameters of the convolution layers (the kernels and bias) 
# are trained such that they were able to identify the patterns in images in the fashionmnist 
# dataset well. Hence, those 'pattern-identification' parameters failing in different types of 
# images is not really a surprise. However, I also note that the layer1 output seems to be not 
# bad, meaning the 'general pattern identification' convolution layer seems to work on different 
# images to some extent too, which also makes sense as the general patterns to be identified 
# would not change much with change in type of objects in images.







