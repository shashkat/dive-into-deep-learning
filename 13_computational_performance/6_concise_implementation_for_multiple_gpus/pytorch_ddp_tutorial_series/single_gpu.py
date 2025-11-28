from numpy import linalg
import torch
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch import nn

# function to load and return dataset (fashionmnist). Images have height and width 28x28, and have 1 channel only
def LoadDatasetFMNIST():
    train_dataset = torchvision.datasets.FashionMNIST(root = '/my_vol/data', train = True, 
        download = False, transform = transforms.ToTensor())
    return (train_dataset)

# trainer class with all necessary methods to train the model
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

# function to create the model instance which can train fashionmnist (using lenet form, as they 
# talked about, in section 13.5 of d2l book)
def CreateModelInstanceFMNIST():
    model_instance = nn.Sequential(
        # shape = (batch_size, 1, 28, 28)
        nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 3), # (batch_size, 20, 26, 26)
        nn.ReLU(),
        nn.AvgPool2d(kernel_size = 2, stride = 2), # (batch_size, 20, 13, 13)
        nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5), # (batch_size, 50, 9, 9)
        nn.ReLU(),
        nn.AvgPool2d(kernel_size = 2, stride = 2), # (batch_size, 50, 4, 4) # missing the rightmost line of pixels in this average pooling due to odd dimensions in image on which doing avgpool
        nn.Flatten(), # (batch_size, 800)
        nn.Linear(in_features = 800, out_features = 128), # (batch_size, 128)
        nn.ReLU(),
        nn.Linear(in_features = 128, out_features = 10) # (batch_size, 10)
    )
    return model_instance

# load training data, model optimizer
def load_train_objs():
    train_set = LoadDatasetFMNIST()  # load your dataset
    model = CreateModelInstanceFMNIST() # load the model we have for fashionmnist
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

# return the dataloader object from the dataset object. 
# This function is keeping pin_memory = True by default
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

# main function
def main(device, total_epochs, save_every, batch_size):
    # device = torch.device('cuda:0')
    # total_epochs = 5
    # save_every = 5
    # batch_size = 128
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)
    # to compare across number of gpus, we don't necessarily need to compute accuracy, as
    # with more gpus, even though the time for a single step would remain roughly same, there
    # will be lesser steps, so total time would be lesser. Also, with more gpus, each step would 
    # be based on more data, so we can take bigger steps, meaning have a higher lr. In any case,
    # the model improves with steps and not num_epochs, so having better quality steps in same 
    # time period is the real benefit with more gpus.

# execute this block only if running this script as the main script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)

# old stuff below





