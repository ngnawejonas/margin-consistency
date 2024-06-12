import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

#@title cifar10_handler
class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
#                    transforms.Normalize(
#                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
#                    transforms.Normalize(
#                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#                    ),
                ]
            )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

# Data.py
def get_CIFAR10(pool_size: int=-1, train: bool=True)->Dataset:
    raw_data = datasets.CIFAR10('./data/CIFAR10', train=train, download=True)
    if pool_size > 0:
        data =  CIFAR10_Handler(raw_data.data[:pool_size], torch.LongTensor(raw_data.targets)[:pool_size], train)
    else:
        data =  CIFAR10_Handler(raw_data.data, torch.LongTensor(raw_data.targets), train)
    return data


class MNIST_Handler(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
          self.transform=transforms.Compose([transforms.ToTensor(),])
        else:
            self.transform=transforms.Compose([transforms.ToTensor(),])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy())
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

def get_MNIST(pool_size: int=-1, train: bool=True)->Dataset:
    raw_data = datasets.MNIST('./data/MNIST', train=train, download=True)
    if pool_size > 0:
        data =  MNIST_Handler(raw_data.data[:pool_size], torch.LongTensor(raw_data.targets)[:pool_size], train)
    else:
        data =  MNIST_Handler(raw_data.data, torch.LongTensor(raw_data.targets), train)
    return data

#@title cifar100_handler
class CIFAR100_Handler(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
#                    transforms.Normalize(
#                        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
#                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
#                    transforms.Normalize(
#                        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
#                    ),
                ]
            )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

# Data.py
def get_CIFAR100(pool_size: int=-1, train: bool=True)->Dataset:
    raw_data = datasets.CIFAR100('./data/CIFAR100', train=train, download=True)
    if pool_size > 0:
        data =  CIFAR100_Handler(raw_data.data[:pool_size], torch.LongTensor(raw_data.targets)[:pool_size], train)
    else:
        data =  CIFAR100_Handler(raw_data.data, torch.LongTensor(raw_data.targets), train)
    return data

def get_ImageNetVal(datapath)->Dataset:
    valdir = os.path.join(datapath, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ]))
    return val_dataset