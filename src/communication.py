from tokenize import group
from networks import LecunNet
import copy
import numpy as np
import torch
import time
import pdb

def initialize_simulator(sim):
    global simulator
    simulator = sim
    return simulator


def send(state, dest, data, receive_callback):
    simulator.schedule(state, dest, receive_callback, data=data)

def wrap_send(state, dest, data, name):
    name += "{" * (48 - len(name))
    name = bytes(name, encoding="utf-8")
    data = name + data
    send(state, dest, data, receive_callback)

def receive_callback(state, src, data):
    _ = yield 0
    buf_key = data[:48].decode("utf-8")
    key_len = buf_key.find("{")
    buf_key = buf_key[:key_len]
    state["receive_buf"][(src, buf_key)] = data[48:]
    yield -1

def wrap_receive(state, src, name):
    while (src, name) not in state["receive_buf"]:
        return 0.001, None
    
    data = state["receive_buf"].pop((src, name))
    return -1, data
    

def wrap_wait(state):
    while state["clockwise_finish"] == False:
        return 0.001
    return -1

def ring_all_reduce(state: dict, src, direction):
    ts = yield 0

    n, prev_id, next_id, group_size, local_id, round_id = state["scheduler_id"], state["prev_id"], state["next_id"], state["group_size"], state["local_id"], state["local_round_id"]

    global_id = simulator.query_peer_state(n)["global_round_id"]

    prev, nxt = [prev_id, next_id], [next_id, prev_id]

    new_net: LecunNet = state["net"]
    
    if group_size != 1:
        new_params = copy.deepcopy(new_net.state_dict())
        param_chunks = []

        types = []
        for param in new_params.values():
            types.append(copy.deepcopy(param.numpy().dtype))

        for param in new_params.values():
            params = list(((param.flatten()) * state["average_completion_time"] / state["completion_time"]).chunk(group_size))
            while len(params) < group_size:
                params.append(torch.Tensor([]))
            param_chunks.append(list(params))
        
        for i in range(group_size - 1):
            # pdb.set_trace()
            cur_id = (direction + local_id + group_size - (1 - 2 * direction) * i) % group_size
            for j in range(len(param_chunks)):
                wrap_send(state, nxt[direction], bytes(param_chunks[j][cur_id].numpy()), "weighta{}_{}_{}_{}_{}".format(direction, round_id, i, j, nxt[direction]))

            cur_id = (cur_id + group_size - (1 - 2 * direction)) % group_size
            j = 0
            while j < len(param_chunks):
                _, buf = wrap_receive(state, prev[direction], "weighta{}_{}_{}_{}_{}".format(direction, round_id, i, j, state["id"]))
                if _ > 0:
                    ts = yield _
                else:
                    buf = torch.Tensor(copy.deepcopy(np.frombuffer(buf, types[j])))
                    param_chunks[j][cur_id] += buf
                    j += 1
        
        if simulator.query_peer_state(n)["global_round_id"] != global_id:
            yield -1

        if direction == 0:
            state["clockwise_finish"] = True
            state["tmp_param_chunks"] = param_chunks
            yield -1
        
        while True:
            _ = wrap_wait(state)
            if _ > 0:
                ts = yield _
            else:
                state["clockwise_finish"] = False
                break

        for i in range(len(param_chunks)):
            param_chunks[i] = (torch.cat(param_chunks[i]) + torch.cat(state["tmp_param_chunks"][i]))

        for i, param in enumerate(new_params.keys()):
            new_params[param] = param_chunks[i].reshape(new_params[param].shape)

        old_params = copy.deepcopy(new_net.state_dict())
        for param in old_params.keys():
            new_params[param] = (new_params[param] - old_params[param] * state["average_completion_time"] / state["completion_time"]) / state["tot_weight"]

        new_net.load_state_dict(new_params)

    elif direction == 0:
        yield -1

    from control import worker_routine
    if ts - state["last_global_ts"] > state["global_round_time"]:
        cur_round = state["local_round_id"]
        cur_id = state["next_id"]
        while cur_id != state["id"]:
            cur_state = simulator.query_peer_state(cur_id)
            cur_round = max(cur_round, cur_state["local_round_id"])
            cur_id = cur_state["next_id"]
        
        state["stop_round"] = cur_round # min(state["stop_round"], cur_round)
        cur_id = state["next_id"]
        while cur_id != state["id"]:
            cur_state = simulator.query_peer_state(cur_id)
            cur_state["stop_round"] = cur_round # min(state["stop_round"], cur_round)
            cur_id = cur_state["next_id"]

    simulator.schedule(state, state["id"], worker_routine, delay=0)
    yield -1

if __name__ == "__main__":
    net = LecunNet()
    ring_all_reduce(None, None, 5, net, net)
    