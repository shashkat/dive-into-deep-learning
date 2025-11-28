from numpy import linalg
import torch
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch import nn

# imports specific to multi-gpu version
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# assign values to some environment variables (through os) related to the main process (rank 0), 
# and call init_process_group function to initialize the process group we have
def ddp_setup():
    # dont need to specify the evironment variables explicitly as torchrun initializes 
    # them automatically. We also don't need to supply rank and world_size to the 
    # init_process_group function, as torchrun handles that too.
    init_process_group(backend = 'nccl')

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
        save_every: int, 
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            print('Snapshot path exists! Loading snapshot...')
            self._load_snapshot(self.snapshot_path)
        # if the snapshot path exists, self.model contains the model from the snapshot, else it 
        # contains the model provided to the __init__ method. In any case, we still want to wrap 
        # DDP around the model
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model = snapshot["MODEL_STATE"]
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f'Resuming training from epoch {self.epochs_run}...')

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

    def _save_snapshot(self, epoch, snapshot_path):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, self.snapshot_path)

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
        shuffle=False,
        sampler=DistributedSampler(dataset) # I believe that in each process, the rank (and 
        # hence which subset of data to take) is known by the calling of init_process_group at 
        # the start. DistributedSampler likely knows to extract information about rank of process
        # from that.
    )

# main function
def main(total_epochs, save_every, batch_size, snapshot_path):
    # total_epochs = 5
    # save_every = 5
    # batch_size = 128
    ddp_setup() # we setup the ddp processes by calling the ddp_setup function here
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group() # once the training is done on each process, we call destroy_process_group() to cleanly exit the multi-process training
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

    # while using torchrun, we just need to call the main function without needing mp.spawn()
    # and torchrun handles the initialization of multiple processes and calling this function 
    # from each of the processes.
    main(args.total_epochs, args.save_every, args.batch_size, snapshot_path = 'snapshot.pt')

# will run this file by calling $ torchrun --standalone --nproc_per_node=gpu multi_gpu_torchrun.py 10 2 
# specifying gpu in nproc_per_node indicates to use all available gpus in each node

# CONCLUSION: I was able to run this file with torchrun, however when I tried to simulate 
# failure in a process using keyboard interrupt and again run the script using the same command,
# it did not work, and gave the error: AttributeError: 'collections.OrderedDict' object has no 
# attribute 'named_parameters', when it was trying to wrap the model around DDP. I think this 
# is because I am using a much more complicated model than was used in the tutorial, and to be 
# able to save this model properly, we would need to save other things also apart from model's 
# state_dict and epoch. I am not investigating this in more detail right now as I think its not 
# required as of now.









