import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import pdb
from networks import LecunNet
import copy
import numpy as np
import random

def cifar10(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainset, testset, trainloader, testloader

def mnist(batch_size):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor()]
    )

    trainset = torchvision.datasets.MNIST(root='mnist_data', train=True, 
                                transform=transform, download=True)

    testset = torchvision.datasets.MNIST(root='mnist_data', 
                                train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                            batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(dataset=testset, 
                            batch_size=batch_size, shuffle=False)
    return trainset, testset, trainloader, testloader

def index_sampler(dataset: torchvision.datasets, ratio: float, batch_size: int):
    tot_datapoints = len(dataset)
    num_samples = int(tot_datapoints * ratio + 0.5)

    indexes = [i for i in range(len(dataset))]
    random.shuffle(indexes)

    res_batched_data = []
    for i in range(len(dataset)):
        tmp_data = []
        for j in range(batch_size):
            tmp_data.append(indexes[i * batch_size + j])
        res_batched_data.append(tmp_data)
        if (i + 1) * batch_size >= num_samples:
            break
    
    return res_batched_data

def data_sampler(dataset: torchvision.datasets, ratio: float, batch_size: int):
    tot_datapoints = len(dataset)
    num_samples = int(tot_datapoints * ratio + 0.5)

    trainloader = DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    res_batched_data = []
    for i, data in enumerate(trainloader):
        res_batched_data.append(data)
        if (i + 1) * batch_size >= num_samples:
            break
    
    return res_batched_data

def load_train_data(dataset: torchvision.datasets):
    batch_size = len(dataset)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_data = 0
    for data in trainloader:
        train_data = data
    return train_data

def load_test_data(dataset: torchvision.datasets, batch_size: int):
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_data = []
    for data in testloader:
        test_data.append(data)
    return test_data

def encode_params(net: LecunNet):
    params = copy.deepcopy(net.state_dict())
    data = b''
    for param in params.values():
        data += bytes(param.flatten().numpy())
    return data

def decode_params(net: LecunNet, data):
    params = net.state_dict()
    offset = 0
    for param in params.keys():
        shape = params[param].shape
        size = params[param].numel()
        dtype = params[param].numpy().dtype
        
        buf = copy.deepcopy(np.frombuffer(data, dtype=dtype, count=size, offset=offset))
        offset += buf[0].itemsize * size

        params[param] = torch.Tensor(buf.reshape(shape))
    
    net.load_state_dict(params)
    return offset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    data = index_sampler(trainset, 0.1, 10)