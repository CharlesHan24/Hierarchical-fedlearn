from typing import List
import heapq
import copy
import pdb
import networkx as nx

BASE_PACKET_OVERHEAD = 100

def s_to_ns(ts):
    return int(ts * 1000000000)

def ns_to_s(s):
    return s / 1000000000
class Simulator(object):
    def __init__(self, init_func: List, topo_delay, shortest_paths, topology: nx.Graph, kwargs):
        n = kwargs["n"]
        self.n = n
        self.state = []
        self.events = [] # heap of [timestamp, callback generator function]
        self.global_time = 0
        self.topo_delay = topo_delay
        self.shortest_paths = shortest_paths
        self.topology = topology
        self.kwargs = kwargs
        self.init_func = init_func

        self.heap_id = 0

        for edge in topology.edges():
            u, v = edge
            topology[u][v]["end_time"] = 0

    def inject_flow(self, x, y, pkt_len, cur_time):
        topology = self.topology
        topology[x][y]["end_time"] = max(topology[x][y]["end_time"], cur_time)
        topology[x][y]["end_time"] += (pkt_len + BASE_PACKET_OVERHEAD) * 1000000 // topology[x][y]["bandwidth"]

    def query_endtime(self, x, y):
        return self.topology[x][y]["end_time"]
        
    def schedule(self, state, dest, callback, delay=0, **kwargs):
        # pdb.set_trace()
        src = state["id"]
        if src != dest:
            path = self.shortest_paths[src][dest]
            self.inject_flow(path[0], path[1], len(kwargs["data"]), self.global_time)
            delay = s_to_ns(self.topo_delay[path[0]][path[1]]) + self.query_endtime(path[0], path[1]) - self.global_time
            path = copy.deepcopy(path)
            path.pop(0)
        else:
            delay = s_to_ns(delay)
            path = []
        
        func = callback(self.state[dest], src, **kwargs)
        next(func)   # the first iteration
        heapq.heappush(self.events, (self.global_time + delay, self.heap_id, path, len(kwargs["data"]) if src != dest else 0, func))
        self.heap_id += 1

    def run(self):
        n = self.n
        for i in range(n):
            self.state.append(dict())
            self.state[-1]["id"] = i
            self.init_func(self.state[-1], self.kwargs)

        exceed = 0
        while self.events != []:
            cur_event = heapq.heappop(self.events)
            self.global_time, _, path, pkt_len, func = cur_event

            # if self.global_time > 10 * 1000000000 and exceed == 0:
            #     print("Aho")
            #     pdb.set_trace()
            #     exceed = 1
            
            if len(path) >= 2:
                self.inject_flow(path[0], path[1], pkt_len, self.global_time)
                delay = s_to_ns(self.topo_delay[path[0]][path[1]]) + self.query_endtime(path[0], path[1]) - self.global_time
                
                path.pop(0)
                heapq.heappush(self.events, (self.global_time + delay, self.heap_id, path, pkt_len, func))
            else:
                ret_ts = func.send(ns_to_s(self.global_time))
                if ret_ts == -1:    # end of function
                    continue
                if self.global_time + s_to_ns(ret_ts) < 0:
                    pdb.set_trace()
                heapq.heappush(self.events, (self.global_time + s_to_ns(ret_ts), self.heap_id, path, pkt_len, func))

            self.heap_id += 1
        pdb.set_trace()
        print("finish")
    
    def query_peer_state(self, peer):
        return self.state[peer]

    def get_time(self):
        return ns_to_s(self.global_time)