from networks import LecunNet
from torch import optim
import torch
import random
import pdb
import copy

EXCHANGE_DENSITY = 2

def rand_exchange(sampled_data, batch_size):
    for i in range(len(sampled_data) * EXCHANGE_DENSITY):
        batch1 = random.randint(0, len(sampled_data) - 1)
        batch2 = random.randint(0, len(sampled_data) - 1)
        id1 = random.randint(0, batch_size - 1)
        id2 = random.randint(0, batch_size - 1)
        
        tmp = copy.deepcopy(sampled_data[batch1][id1])
        sampled_data[batch1][id1] = sampled_data[batch2][id2]
        sampled_data[batch2][id2] = tmp


def train(state):
    running_loss = 0.0
    num_examples = 0
    net: LecunNet = state["net"]
    criterion = state["criterion"]
    batch_size = state["batch_size"]
    portion = int(state["train_portion"] * len(state["sampled_data"]) * state["average_completion_time"] / state["completion_time"] + 0.99)
    random.shuffle(state["sampled_data"])
    rand_exchange(state["sampled_data"], batch_size)
    sampled_data = state["sampled_data"][:portion]

    optimizer = optim.SGD(net.parameters(), lr=state["lr"], momentum=0.9)
    print("Training on device #{}. Epoch = {}".format(state["id"], state["local_round_id"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.train()
    for i, data in enumerate(sampled_data):
        # get the inputs; data is a list of [inputs, labels]
        inputs = []
        labels = []
        for j, idx in enumerate(data):
            inputs.append(state["train_data"][0][idx])
            labels.append(state["train_data"][1][idx])
        inputs = torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        num_examples += len(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    device = torch.device("cpu")
    net.to(device)

    running_loss /= len(sampled_data)
    print("Running loss = {}".format(running_loss))

    return running_loss

def eval(state):
    net = state["net"]
    criterion = state["criterion"]
    test_data = state["test_data"]

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
