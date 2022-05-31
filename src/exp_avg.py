import torch
import torchvision
import torchvision.transforms as transforms
from utilities import data_sampler
from networks import VGG, LecunNet, LecunNet_Bigger
from torch import nn
from torch import optim
import copy
import argparse
import json
import pdb
import time
from utilities import cifar10, mnist

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True, type=str)
args = parser.parse_args()

fin = open(args.config_file, "r")
config = json.load(fin)

batch_size = config["batch_size"]
parallel_degree = config["n"]
portion = config["portion"] * config["ratio"]
lr = config["lr"]

if config["dataset"] == "mnist":
    trainset, testset, trainloader, testloader = mnist(batch_size)
else:
    trainset, testset, trainloader, testloader = cifar10(batch_size)

if config["net"] == "LecunNet":
    net_class = LecunNet
    netarg = {"in_channel": 1 if config["dataset"] == "mnist" else 3}
elif config["net"] == "LecunNet_Bigger":
    net_class = LecunNet_Bigger
    netarg = {"in_channel": 1 if config["dataset"] == "mnist" else 3}
else:
    net_class = VGG
    netarg = {"vgg_name": "VGG13"}

def train(net, train_data, testloader, lr=0.003):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for i, data in enumerate(train_data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % (len(train_data) // 10) == 0:
            print(eval(single_thread_net, testloader))

def eval(net, testloader):
    criterion = nn.CrossEntropyLoss()
    test_data = testloader

    net.eval()

    cumulative_loss = 0.0
    correct = 0
    num_examples = 0
    num_batch = 0
    for i, data in enumerate(test_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        num_examples += len(inputs)
        num_batch += 1

        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        cumulative_loss += loss.item()

        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum().item()
        
    cumulative_loss /= num_batch
    correct /= num_examples

    return cumulative_loss, correct

net = net_class(**netarg)
# for i in range(2):
#     complete_data = data_sampler(trainset, 1.0, batch_size)

#     train(net, complete_data)
#     print(eval(net, testloader))

torch.save(net.state_dict(), "./1epoch.pt")
net.load_state_dict(torch.load("./1epoch.pt"))
initial_net = copy.deepcopy(net)

avg_step_test_cases = [25]


complete_data = data_sampler(trainset, 0.8, batch_size)
single_thread_net = copy.deepcopy(net)
x = time.time()

for i in range(20):
    train(single_thread_net, complete_data, lr=lr, testloader=testloader)
    print("original:", eval(single_thread_net, testloader))
print(time.time() - x)

lr *= 10
# batch_size //= 10
for step in avg_step_test_cases:
    net = copy.deepcopy(net_class(**netarg))
    rounds = 5
    print(rounds)
    for i in range(rounds * parallel_degree * 2):
        # pdb.set_trace()
        dist_net = []
        for j in range(parallel_degree):
            dist_net.append(copy.deepcopy(net))
            num_samples = portion * len(trainset)
            cur_data = data_sampler(trainset, portion, batch_size)
            train(dist_net[j], cur_data, lr=lr)
        
        print(i, "aggregating...")

        param_set = []
        for j in range(parallel_degree):
            param_set.append(dist_net[j].state_dict())

        for param in param_set[0].keys():
            for j in range(1, parallel_degree, 1):
                param_set[0][param] += param_set[j][param]
            
            param_set[0][param] /= parallel_degree

        net.load_state_dict(param_set[0])
        if i % 1 == 0:
            print(eval(net, testloader))
    print("step = {}: ".format(step), eval(net, testloader))
