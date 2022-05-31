from utilities import index_sampler, load_test_data, encode_params, decode_params, load_train_data
from train_and_eval import train, eval
from communication import ring_all_reduce
from networks import LecunNet
from communication import send
import random
import torch
import copy
import json
from policy import policy_continuous_optimization, policy_fedavg, policy_random_search, policy_fedavg_ring, policy_fedavg_partial
import pdb
from simulator import Simulator
import math

INF = 1000000000

def initialize_simulator(sim):
    global simulator
    simulator = sim
    return simulator


def initialize_all(state: dict, kwargs):
    n = kwargs["n"]
    state["log"] = kwargs["log"]
    state["topology"] = kwargs["topology"]
    state["n"] = n
    state["scheduler_id"] = n - 1

    state["net"] = copy.deepcopy(kwargs["net"])
    # if state["id"] != 0:
    #     peer_net: LecunNet = simulator.query_peer_state(0)["net"]
    #     weights = copy.deepcopy(peer_net.state_dict())
    #     state["net"].load_state_dict(weights)
    
    state["criterion"] = torch.nn.CrossEntropyLoss()

    train_dataset = kwargs["train_dataset"]
    state["sampled_data"] = index_sampler(train_dataset, kwargs["ratio"], kwargs["batch_size"])
    
    test_dataset = kwargs["test_dataset"]
    if state["id"] == 0:
        state["train_data"] = load_train_data(train_dataset)
        state["test_data"] = load_test_data(test_dataset, kwargs["batch_size"])
    else:
        peer_data = simulator.query_peer_state(0)["test_data"]
        state["test_data"] = peer_data
        peer_data = simulator.query_peer_state(0)["train_data"]
        state["train_data"] = peer_data
    
    state["tot_rounds"] = kwargs["tot_rounds"]
    state["last_global_ts"] = 0
    state["stop_round"] = INF
    state["local_round_id"] = 0

    # kwargs["average_completion_time"] refers to the time it takes to train on the whole dataset for one epoch
    state["batch_size"] = kwargs["batch_size"]
    state["lr"] = kwargs["lr"]
    state["average_completion_time"] = kwargs["average_completion_time"] * kwargs["ratio"] * kwargs["portion"]

    # state["completion_time"] = (random.random() * 0.6 + 0.7) * state["average_completion_time"]
    state["completion_time"] = state["average_completion_time"] * math.exp(random.gauss(0, 0.5) - 0.2)
    state["train_portion"] = kwargs["portion"] * (1 if "portion_factor" not in kwargs else kwargs["portion_factor"]) # 0 < portion <= 1
    state["global_round_time"] = kwargs["global_round_time"]
    state["receive_buf"] = dict()
    if state["id"] == n - 1:
        state["delay"] = kwargs["delay"]
        state["ngroup"] = kwargs["ngroup"]
        state["correct_ratio"] = kwargs["correct_ratio"]
        state["bottleneck_group_num"] = kwargs["bottleneck_group_num"]
    
    if state["id"] < n - 1:
        state["latest_running_loss"] = 1.
        state["clockwise_finish"]= False
        # simulator.schedule(state, state["id"], worker_routine, delay=0)
        pass
    else:
        # scheduler
        state["running_loss"] = [1 for i in range(n - 1)]
        state["aggre_net"] = [copy.deepcopy(state["net"]) for i in range(n - 1)]
        state["received"] = 0
        state["completion_time"] = [simulator.query_peer_state(i)["completion_time"] for i in range(n - 1)]
        state["global_round_id"] = 0
        state["decision_dict"] = []
        for i in range(n - 1):
            state["decision_dict"].append({"local_id": 0})
        state["tot_group"] = n - 1

        if kwargs["policy"] == "fedavg":
            state["policy"] = policy_fedavg
        elif kwargs["policy"] == "random_search":
            state["policy"] = policy_random_search
        elif kwargs["policy"] == "continuous_optimization":
            state["policy"] = policy_continuous_optimization
        elif kwargs["policy"] == "ring_all_reduce":
            state["policy"] = policy_fedavg_ring
        elif kwargs["policy"] == "fedavg_partial":
            state["policy"] = policy_fedavg_partial


        simulator.schedule(state, state["id"], scheduler_routine, delay=0)


def worker_callback(state, src, data):
    # pdb.set_trace()
    assert(src == state["n"] - 1)
    ts = yield 0

    offset = decode_params(state["net"], data)
    data = data[offset:]
    data = json.loads(data.decode("ascii").replace("'", "\""))

    for item in data.keys():
        state[item] = data[item]

    state["last_global_ts"] = ts
    state["stop_round"] = INF
    state["local_round_id"] = 0
    simulator.schedule(state, state["id"], worker_routine, 0)

    controller_state = simulator.query_peer_state(src)
    x = controller_state["decision_dict"][state["id"]]["next_id"]
    while x != state["id"]:
        new_state = simulator.query_peer_state(x)
        new_state["last_global_ts"] = ts
        new_state["stop_round"] = INF
        new_state["local_round_id"] = 0
        new_state["net"] = copy.deepcopy(state["net"])
        for item in controller_state["decision_dict"][x]:
            new_state[item] = controller_state["decision_dict"][x][item]
        simulator.schedule(new_state, x, worker_routine, 0)
        x = controller_state["decision_dict"][x]["next_id"]

    yield -1

def worker_wrap_send(state, dest, running_loss):
    extra_dict = dict()
    extra_dict["running_loss"] = running_loss
    extra_dict = str(extra_dict).encode("ascii")

    data = encode_params(state["net"]) + extra_dict
    send(state, dest, data, scheduler_callback)


def worker_routine(state: dict, src):
    ts = yield 0
    if state["stop_round"] > state["local_round_id"]:
        if state["id"] == 0:
            print("SSS important: {}".format(simulator.global_time))
        state["local_round_id"] += 1
        state["latest_running_loss"] = train(state)
        for i in range(2):
            simulator.schedule(state, state["id"], ring_all_reduce, delay=state["average_completion_time"], direction=i)
        yield -1
    
    if state["local_id"] == 0:
        worker_wrap_send(state, state["n"] - 1, state["latest_running_loss"])
    yield -1

def scheduler_callback(state: dict, src, data):
    ts = yield 0

    offset = decode_params(state["aggre_net"][src], data)
    data = data[offset:]
    data = json.loads(data.decode("ascii").replace("'", "\""))
    state["running_loss"][src] = data["running_loss"]
    state["received"] += 1
    if state["received"] == state["tot_group"]:
        simulator.schedule(state, state["id"], scheduler_routine, delay=0)
    yield -1
    
def wrap_scheduler_send(state: dict, dest, decision_dict):
    decision_dict = str(decision_dict).encode("ascii")
    data = encode_params(state["net"]) + decision_dict
    send(state, dest, data, worker_callback)

def scheduler_routine(state: dict, src):
    ts = yield 0

    import pdb

    state["received"] = 0
    state["global_round_id"] += 1
    # The simplist aggregation approach is weighted aggregation according to the size of each group
    # pdb.set_trace()
    params = None
    n = state["n"]

    if state["global_round_id"] > 1:
        # pdb.set_trace()
        divisor = 0

        for i in range(0, n - 1):
            if state["decision_dict"][i]["local_id"] == 0:
                if params == None:
                    params = copy.deepcopy(state["aggre_net"][i].state_dict())
                    for param in params.keys():
                        params[param] *= state["decision_dict"][i]["group_size"] * (simulator.query_peer_state(i)["local_round_id"])
                else:
                    new_params = copy.deepcopy(state["aggre_net"][i].state_dict())
                    for param in params.keys():
                        params[param] += new_params[param] * state["decision_dict"][i]["group_size"] * (simulator.query_peer_state(i)["local_round_id"])
                divisor += state["decision_dict"][i]["group_size"] * (simulator.query_peer_state(i)["local_round_id"])

        for param in params.values():
            param /= divisor
        
        state["net"].load_state_dict(params)

    running_loss = []
    for i in range(n - 1):
        running_loss.append(simulator.query_peer_state(i)["latest_running_loss"])
    if state["policy"] != policy_continuous_optimization or state["global_round_id"] == 1:
        decision_dict, tot_group = state["policy"](state, running_loss, state["delay"])
    else:
        decision_dict, tot_group = state["decision_dict"], state["tot_group"]
    state["decision_dict"] = decision_dict
    state["tot_group"] = tot_group

    _, correct_ratio = eval(state)
    print("Time: {}. At {}th round, lr = {}, loss = {}, correct_ratio = {}".format(simulator.get_time(), state["global_round_id"], state["lr"], _, correct_ratio))
    state["log"].write("Time: {}. At {}th round, lr = {}, loss = {}, correct_ratio = {}".format(simulator.get_time(), state["global_round_id"], state["lr"], _, correct_ratio))
    state["log"].flush()
    if correct_ratio > state["correct_ratio"] or state["global_round_id"] > state["tot_rounds"]:
        pdb.set_trace()
        yield -1
    
    if state["global_round_id"] > 1:
        for i in range(n):
            if simulator.query_peer_state(i)["lr"] > 0.001:
                simulator.query_peer_state(i)["lr"] *= 0.9 ** state["global_round_time"]

    for i in range(n - 1):
        if state["decision_dict"][i]["local_id"] == 0:
            wrap_scheduler_send(state, i, decision_dict[i])
    yield -1
    