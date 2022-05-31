import pickle
from simulator import Simulator
from control import initialize_all
import control
import communication
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import pdb
import json
import argparse
from utilities import mnist, cifar10
from networks import LecunNet, LecunNet_Bigger, VGG
import random
from topology import Topology

def tree_topo_20():
    graph = nx.Graph()
    for i in range(4):
        for j in range(4):
            graph.add_edge(i * 4 + j, 16 + i, weight=1)
    
    for i in range(4):
        graph.add_edge(16 + i, 20, weight=5)
    
    res = dict(nx.all_pairs_dijkstra_path_length(graph))
    for key1 in res.keys():
        for key2 in res[key1].keys():
            res[key1][key2] *= 0.005
    return res

def tree_topo_10():
    graph = nx.Graph()
    for i in range(2):
        for j in range(4):
            graph.add_edge(i * 4 + j, 8 + i, weight=1)
    for i in range(2):
        graph.add_edge(8 + i, 10, weight=5)
    res = dict(nx.all_pairs_dijkstra_path_length(graph))
    for key1 in res.keys():
        for key2 in res[key1].keys():
            res[key1][key2] *= 0.01
    print(res)
    return res

if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    args = parser.parse_args()

    fin = open(args.config_file, "r")
    kwargs = json.load(fin)
    kwargs["n"] += 1

    n = kwargs["n"]

    # kwargs["tot_rounds"] = 100
    batch_size = kwargs["batch_size"]
    
    if kwargs["dataset"] == "mnist":
        trainset, testset, _, _ = mnist(batch_size)
    else:
        trainset, testset, _, _ = cifar10(batch_size)

    kwargs["train_dataset"] = trainset
    kwargs["test_dataset"] = testset

    if kwargs["net"] == "LecunNet":
        kwargs["net"] = LecunNet(1 if kwargs["dataset"] == "mnist" else 3)
    
    elif kwargs["net"] == "LecunNet_Bigger":
        kwargs["net"] = LecunNet_Bigger(1 if kwargs["dataset"] == "mnist" else 3)
    else:
        kwargs["net"] = VGG("VGG13")
    # kwargs["average_completion_time"] = 30
    # kwargs["ratio"] = 0.2
    # kwargs["portion"] = 0.5          # ~1.5s for training and ~1.5s for communication
    # kwargs["correct_ratio"] = 0.63
    # kwargs["ngroup"] = 1
    
    # kwargs["global_round_time"] = 10 # 10s for each round

    with open("configs/" + kwargs["topology"], "rb") as f:
        topology: Topology = pickle.load(f)
    kwargs["topology"] = topology
    kwargs["delay"] = topology.delay

    kwargs["log"] = open(kwargs["log"], "w")

    sim = Simulator(initialize_all, kwargs["delay"], topology.shortest_paths, topology.topology, kwargs)
    sim = control.initialize_simulator(sim)
    sim = communication.initialize_simulator(sim)

    sim.run()