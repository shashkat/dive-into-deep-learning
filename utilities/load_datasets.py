import torchvision
from torchvision import transforms

# FOR LOCAL
# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 1 channel only
def LoadDatasetFMNIST():
    train_dataset = torchvision.datasets.FashionMNIST(root = '/Users/shashankkatiyar/Documents/learning_ml/dive-into-deep-learning/data', 
    	train = True, download = False, transform = transforms.ToTensor())
    return (train_dataset)

# FOR MODAL INSTANCE
# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 1 channel only
def LoadDatasetFMNIST():
    train_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', train = True, 
        download = False, transform = transforms.ToTensor())
    return (train_dataset)

# cifar10 for modal instance
# function to load and return dataset (cifar10). Images have height and width 32x32, and have 
# 3 channels
def LoadDatasetCIFAR10(train = True):
    train_dataset = torchvision.datasets.CIFAR10(root = '/my_vol/data', 
    	train = train, download = False, transform = transforms.ToTensor())
    return (train_dataset)