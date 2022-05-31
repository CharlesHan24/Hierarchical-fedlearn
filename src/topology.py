import networkx as nx
import pickle
import copy
import random
import pdb
from networkx.algorithms.community.centrality import girvan_newman

def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight="weight")
    return max(centrality, key=centrality.get)

class Topology(object):
    def __init__(self, graph: nx.Graph, delay: dict, shortest_paths: dict, clusters: list):
        self.topology = graph
        self.delay = delay
        self.shortest_paths = shortest_paths
        self.clusters = clusters
        self.n = graph.number_of_nodes()


def fat_tree(k):
    graph = nx.Graph()

    for i in range(k * k // 4):
        core_sw = i + 1
        for j in range(k):
            aggre_sw = k * k // 4 + j * (k // 2) + i // (k // 2) + 1
            graph.add_edge(core_sw - 1, aggre_sw - 1, weight=0.05, bandwidth=100 * 1000)

    for i in range(k):
        for j1 in range(k // 2):
            for j2 in range(k // 2):
                aggre_sw = k * k // 4 + (i) * (k // 2) + j1 + 1
                tor_sw = k * k // 4 + k * (k // 2) + (i) * (k // 2) + j2 + 1
                graph.add_edge(aggre_sw - 1, tor_sw - 1, weight=0.01, bandwidth= 30 * 1000)
    
    n = graph.number_of_nodes()
    for i in range(k * k // 4):
        graph.add_edge(n, i, weight=0.05, bandwidth=10 * 1000)
    return graph

def variant_fat_tree(k):
    graph = nx.Graph()
    for i in range(1):
        core_sw = i + 1
        for j in range(k):
            aggre_sw = 1 + j * (k // 2) + i + 1
            graph.add_edge(core_sw - 1, aggre_sw - 1, weight=0.05, bandwidth=10000 * 1000)

    for i in range(k):
        for j1 in range(k // 2):
            for j2 in range(k // 2):
                aggre_sw = 1 + (i) * (k // 2) + j1 + 1
                tor_sw = 1 + k * (k // 2) + (i) * (k // 2) + j2 + 1
                graph.add_edge(aggre_sw - 1, tor_sw - 1, weight=0.01, bandwidth=100 * 1000)
    
    n = graph.number_of_nodes()
    for i in range(1):
        graph.add_edge(n, i, weight=0.05, bandwidth = 100 * 1000)
    return graph

def dump_graph_with_path(file_name, graph, k):
    delay = dict(nx.all_pairs_dijkstra_path_length(graph))

    pdb.set_trace()

    eps = 1e-7
    tot_pool = 100
    shortest_paths_list = []
    for i in range(tot_pool):
        new_graph = copy.deepcopy(graph)
        for edge in new_graph.edges():
            new_graph[edge[0]][edge[1]]["weight"] += eps * (random.random() - 0.5)
        shortest_paths_list.append(dict(nx.all_pairs_dijkstra_path(new_graph)))
    
    shortest_paths = copy.deepcopy(shortest_paths_list[0])
    n = graph.number_of_nodes()
    for i in range(n):
        for j in range(n):
            shortest_paths[i][j] = shortest_paths_list[random.randint(0, tot_pool - 1)][i][j]

    clusters = list(girvan_newman(graph, most_central_edge))

    for j1 in range(1):
        for i in range(k):
            for j in range(i + 1, k, 1):
                sw1 = 1 + (i) * (k // 2) + 1 + j1
                sw2 = 1 + (j) * (k // 2) + 1 + j1
                graph.add_edge(sw1 - 1, sw2 - 1, weight=0.06 + 0.04 * (j - i) / (k - 1), bandwidth=10000 * 1000)

                sw1 = 1 + (i) * (k // 2) + 1 + j1 + k * (k // 2)
                sw2 = 1 + (j) * (k // 2) + 1 + j1 + k * (k // 2)
                graph.add_edge(sw1 - 1, sw2 - 1, weight=0.06 + 0.04 * (j - i) / (k - 1), bandwidth=10000 * 1000)

    result = Topology(graph, delay, shortest_paths, clusters)
    with open(file_name, "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    dump_graph_with_path("configs/var_fat_tree_10.gt", variant_fat_tree(10), 10)
