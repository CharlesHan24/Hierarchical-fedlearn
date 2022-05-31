from os.path import exists
import random
from tokenize import group
from typing import List
import pdb
import numpy
import torch
import torch.nn.functional as torchfunc
from torch import optim
import torch.nn as nn
import numpy as np
import pickle
import copy

INF = 1000000007

def build_decision_dict(n, group_num, delay, best_group_id, best_next, state):
    t = state["completion_time"]
    bar_t = state["average_completion_time"]

    group_bin = [[] for i in range(group_num)]
    for i in range(n):
        group_bin[best_group_id[i]].append(i)
        
    decision_dict = [dict() for i in range(n)]
    for i in range(n):
        decision_dict[i]["group_size"] = len(group_bin[best_group_id[i]])
        decision_dict[i]["next_id"] = best_next[i]
        decision_dict[best_next[i]]["prev_id"] = i
    
    tot_group = 0
    for i in range(group_num):
        if len(group_bin[i]) != 0:
            tot_group += 1
            x = [group_bin[i][0]]
            for j in range(1, len(group_bin[i])):
                if delay[n][x[0]] > delay[n][group_bin[i][j]] + 1e-7:
                    x = [group_bin[i][j]]
                elif delay[n][x[0]] > delay[n][group_bin[i][j]] - 1e-7:
                    x.append(group_bin[i][j])

            x = x[random.randint(0, len(x) - 1)]

            tot_weight = 0
            for j in range(len(group_bin[i])):
                tot_weight += bar_t / t[group_bin[i][j]]

            for j in range(len(group_bin[i])):
                decision_dict[x]["local_id"] = j
                decision_dict[x]["tot_weight"] = tot_weight
                x = best_next[x]
    
    return decision_dict, tot_group

def greedy_cycle(n, group_num, grouping, delay):
    group_bin = [[] for i in range(group_num)]
    eps = 1e-9
    for i in range(n):
        group_bin[grouping[i]].append(i)
    
    cycle = [0 for i in range(n)]
    visited = [0 for i in range(n)]
    for i in range(group_num):
        if len(group_bin[i]) == 0:
            continue
        x = group_bin[i][0]
        visited[x] = 1
        tot = len(group_bin[i])
        for j in range(tot - 1):
            min_delay = INF
            min_id = 0
            for k in range(tot):
                if not visited[group_bin[i][k]]:
                    if min_delay > delay[x][group_bin[i][k]]:
                        min_id = group_bin[i][k]
                        min_delay = delay[x][group_bin[i][k]]
            cycle[x] = min_id
            x = min_id
            visited[x] = 1
        cycle[x] = group_bin[i][0]
    
    return cycle

def policy_fedavg_ring(state, avg_loss: List[float], delay):
    n = state["n"] - 1
    
    decision_dict = [dict() for i in range(n)]
    comm_order = [i for i in range(n)]
    random.shuffle(comm_order)

    grouping = []
    for i in range(n):
        grouping.append(0)
    
    cycle = greedy_cycle(n, 1, grouping, delay)
    return build_decision_dict(n, 1, delay, grouping, cycle, state)

    # for i in range(n):
    #     idx = comm_order[i]
    #     decision_dict[idx]["local_id"] = i

    #     decision_dict[idx]["next_id"] = comm_order[(i + 1) % n]
    #     decision_dict[idx]["prev_id"] = comm_order[(i + n - 1) % n]
    #     decision_dict[idx]["group_size"] = n

    
    return decision_dict, 1

def policy_fedavg(state, avg_loss: List[float], delay):
    n = state["n"] - 1
    decision_dict = [dict() for i in range(n)]
    t = state["completion_time"]
    bar_t = state["average_completion_time"]

    tot_weight = 0

    for i in range(n):
        decision_dict[i]["local_id"] = 0

        decision_dict[i]["next_id"] = i
        decision_dict[i]["prev_id"] = i
        decision_dict[i]["group_size"] = 1
        tot_weight += bar_t / t[i]
    
    for i in range(n):
        decision_dict[i]["tot_weight"] = tot_weight
    
    return decision_dict, n

def policy_fedavg_partial(state, avg_loss: List[float], delay):
    n = state["n"] - 1
    decision_dict = [dict() for i in range(n)]
    t = state["completion_time"]
    bar_t = state["average_completion_time"]

    tot_weight = 0

    tot_group = n // 10
    sampled = random.sample([i for i in range(n)], tot_group)

    for i in range(n):
        decision_dict[i]["local_id"] = 10000

        decision_dict[i]["next_id"] = i
        decision_dict[i]["prev_id"] = i
        decision_dict[i]["group_size"] = 1
    
    for x in sampled:
        decision_dict[x]["local_id"] = 0

        decision_dict[x]["next_id"] = x
        decision_dict[x]["prev_id"] = x
        decision_dict[x]["group_size"] = 1
        tot_weight += bar_t / t[x]
    
    for i in range(n):
        decision_dict[i]["tot_weight"] = tot_weight
    
    return decision_dict, tot_group

def calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t):
    # pdb.set_trace()
    group_bin = [[] for i in range(group_num)]
    eps = 1e-9
    for i in range(n):
        group_bin[grouping[i]].append(i)
    tot_loss = 0
    # pdb.set_trace()
    for i in range(group_num):
        divisor = 0.2
        # for j in range(len(group_bin[i])):
        #     divisor = max(divisor, t[group_bin[i][j]])
        for j in range(len(group_bin[i])):
            x = group_bin[i][j]
            divisor += delay[x][cycle[x]]
        for j in range(len(group_bin[i])):
            tot_loss += avg_loss[group_bin[i][j]] / t[group_bin[i][j]] / (divisor + eps)
    return tot_loss


def best_random_cycle(n, group_num, grouping, delay):
    group_bin = [[] for i in range(group_num)]
    eps = 1e-9
    for i in range(n):
        group_bin[grouping[i]].append(i)
    
    cycle = [0 for i in range(n)]
    current_cycle = [0 for i in range(n)]
    visited = [0 for i in range(n)]
    for i in range(group_num):
        if len(group_bin[i]) == 0:
            continue
        
        min_tot_delay = INF
        for _ in range(30):
            tot_delay = 0

            tot = len(group_bin[i])
            order = [j for j in range(tot)]
            random.shuffle(order)
            x = group_bin[i][order[0]]
            visited[x] = 1

            for j in range(tot - 1):
                current_cycle[x] = group_bin[i][order[j + 1]]
                tot_delay += delay[x][group_bin[i][order[j + 1]]]
                x = group_bin[i][order[j + 1]]
                visited[x] = 1
            current_cycle[x] = group_bin[i][order[0]]
            tot_delay += delay[x][group_bin[i][order[0]]]
            for j in range(tot):
                visited[group_bin[i][j]] = 0
            if tot_delay < min_tot_delay:
                min_tot_delay = tot_delay
                for j in range(tot):
                    cycle[group_bin[i][j]] = current_cycle[group_bin[i][j]]
    
    return cycle


def policy_random_search(state, avg_loss: List[float], delay):
    n = state["n"] - 1
    t = state["completion_time"]
    group_num = state["bottleneck_group_num"]

    best_next = [0 for i in range(n)]
    best_group_id = [0 for i in range(n)]
    max_value = 0

    for i in range(2000):
        if i % 100 == 0:
            print(i)
        grouping = []
        for j in range(n):
            grouping.append(random.randint(0, group_num - 1))
        
        cycle = greedy_cycle(n, group_num, grouping, delay)
        if calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t) > max_value:
            max_value = calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t)
            best_next = cycle
            best_group_id = grouping
        
        cycle = best_random_cycle(n, group_num, grouping, delay)
        if calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t) > max_value:
            max_value = calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t)
            best_next = cycle
            best_group_id = grouping
    

    return build_decision_dict(n, group_num, delay, best_group_id, best_next, state)


def _wrap_calc_loss(n, X, group_num, delay, avg_loss, t):
    grouping = torch.argmax(X, dim=1).numpy()
    cycle = greedy_cycle(n, group_num, grouping, delay)
    return calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t)

def policy_continuous_optimization(state, avg_loss: List[float], delay):
    n = state["n"] - 1
    t = state["completion_time"]
    bar_t = state["average_completion_time"]
    group_num = state["bottleneck_group_num"]

    clusters = state["topology"].clusters
    mnt = []
    for i in range(n):
        cur_mnt = INF
        for j in range(n):
            if i != j:
                cur_mnt = min(cur_mnt, delay[i][j])
        mnt.append(cur_mnt)

    
    
    X = torch.zeros((n, group_num))
    
    # pdb.set_trace()
    # for i, group in enumerate(clusters[group_num - 5]):
    # for i, group in enumerate(clusters[group_num - 2]):
    #     if n in group:
    #         group.remove(n)
    #     for member in group:
    #         X[member][i] = 10
    
    X.requires_grad = True

    # pdb.set_trace()

    class Objective(object):
        def __init__(self, n, group_num, avg_loss, t, delay, clusters, mnt):
            self.n = n
            self.group_num = group_num
            self.avg_loss = torch.Tensor(avg_loss) / torch.Tensor(t) * torch.mean(torch.Tensor(t))
            
            self.delay = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(n):
                    self.delay[i][j] = delay[i][j]
            self.delay = torch.Tensor(self.delay)
            self.clusters = []
            # for group in clusters[group_num - 5]:
            for group in clusters[9]:
                if self.n in group:
                    group.remove(self.n)
                self.clusters.append(torch.tensor(list(group), dtype=torch.int32))
            self.mnt = torch.Tensor(mnt)

        def forward(self, X: torch.Tensor):
            # pdb.set_trace()
            # Y = torchfunc.one_hot(XX, num_classes=self.group_num).type(torch.float32)
            
            Y = torchfunc.softmax(X, dim=1) # X / torch.sum(X, dim=1, keepdim=True) # torchfunc.softmax(X, dim=1)
            Z = Y / torch.amax(torchfunc.softmax(X, dim=1), dim=1).unsqueeze(1)
            # Z = Z ** 2
            # pdb.set_trace()
            # Y = 1.0 * (X > torch.amax(X, dim=1).unsqueeze(1) - 1e-5)
            numerators = torch.matmul(self.avg_loss, Y)#torch.maximum(Z ** 3, 0.9 * Z))

            minor_delay = torch.matmul(self.mnt, Y) / 2#Z)

            for i in range(self.group_num):
                Z_i = Z[:, i].flatten()
                tmp = torch.amax(Z_i.unsqueeze(0) * self.delay, dim=1)
                tmp = tmp * Z_i
                major_delay = 0
                for group in self.clusters:
                    to_max = torch.max(torch.index_select(tmp, dim=0, index=group))
                    major_delay += to_max
                minor_delay[i] += major_delay
            denominator = minor_delay + 0.2
            loss = torch.sum(numerators / denominator)

            return -loss, Z, Y
        
    loss_func = Objective(n, group_num, avg_loss, t, delay, clusters, mnt)
    loss, _1, _2 = loss_func.forward(X)
    print(_wrap_calc_loss(n, X, group_num, delay, avg_loss, t))
    fout = open("loss.txt", "w")
    fout.write("{} {}\n".format(_wrap_calc_loss(n, X, group_num, delay, avg_loss, t), loss))

    X = torch.randn((n, group_num), requires_grad=True)

    lr = 0.1
    optimizer = optim.Adam([X], lr=lr)# SGD([X], lr=0.1, momentum=0.8)
    steps = 10000 if not exists("grouping6.txt") else 0

    noise_factor = 3

    last_loss = 0
    last_X = copy.deepcopy(X)

    # pdb.set_trace()
    

    for i in range(steps):
        # pdb.set_trace()
        if i % 50 == 0:
            fout.write("{} {}\n".format(_wrap_calc_loss(n, X, group_num, delay, avg_loss, t), loss))
            print(i)
            if i > 5000:
                for g in optimizer.param_groups:
                    lr *= 0.99
                    if lr < 0.001:
                        lr = 0.001
                    g["lr"] = lr

            if noise_factor < 10:
                noise_factor *= 1 / 0.99
            else:
                noise_factor *= 1 / 0.99
            print(loss)
        # if i % 50 == 1:
        #     noise_factor /= 5
        # if i % 50 == 2:
        #     noise_factor *= 5

        optimizer.zero_grad()
        loss, _1, _2 = loss_func.forward(X)
        # pdb.set_trace()
        loss.backward()
    
        optimizer.step()

        # X.requires_grad = False
        # # pdb.set_trace()
        if i % 200 == 0:
            X.requires_grad = False
            print(_wrap_calc_loss(n, X, group_num, delay, avg_loss, t))
            # pdb.set_trace()
            if i > 5000:
                if loss > last_loss:
                    X = copy.deepcopy(last_X)
                    loss = last_loss

            last_loss = loss
            last_X = copy.deepcopy(X)
            # pdb.set_trace()
            X = X + torch.std(torch.abs(X)) * torch.randn(X.shape) / noise_factor
            
            X.requires_grad = True
            optimizer = optim.Adam([X], lr=lr)# SGD([X], lr=0.1, momentum=0.8)
            

        # elif i % 50 == 0:
        #     X = X + torch.std(torch.abs(X)) * torch.randn(X.shape) / noise_factor
        # X.requires_grad = True
    # pdb.set_trace()
    grouping = torch.argmax(X, dim=1).numpy()
    tmp = [[] for i in range(group_num)]
    for i in range(n):
        tmp[grouping[i]].append(t[i])
    for i in range(group_num):
        fout.write("{} {}\n".format(sum(tmp[i]) / len(tmp[i]), len(tmp[i])))
    
    best_next = [0 for i in range(n)]
    best_group_id = [0 for i in range(n)]
    max_value = 0

    grouping = torch.argmax(X, dim=1).numpy()
    if exists("grouping6.txt"):
        with open("grouping6.txt", "rb") as f:
            grouping = pickle.load(f)
    with open("grouping6.txt", "wb") as f:
        pickle.dump(grouping, f)
        f.flush()

    cycle = greedy_cycle(n, group_num, grouping, delay)
    if calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t) > max_value:
        max_value = calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t)
        best_next = cycle
        best_group_id = grouping

    cycle = best_random_cycle(n, group_num, grouping, delay)
    if calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t) > max_value:
        max_value = calc_loss(n, group_num, grouping, cycle, delay, avg_loss, t)
        best_next = cycle
        best_group_id = grouping

    return build_decision_dict(n, group_num, delay, best_group_id, best_next, state)
