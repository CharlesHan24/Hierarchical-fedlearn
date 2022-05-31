# Hierarchical Aggregation for Efficient Federated Learning at Scale

This repository contains the source code of my PKU undergraduate thesis project. 

## How to run

To reproduce the experimental results, first we need to generate the topology of the virtual WAN by running `topology.py`:

```
python3 topology.py
```

We can then launch all the experiments using our provided configurations or your custom configurations. You can see example configurations within the folder `configs/`.

To start the simulation configured by the file `cfg`, we simply run

```
python3 run.py --config_file cfg
```

The program will automatically simulate our hierarchical federated learning process, keeping track of the evaluation accuracy and loss value with respect to the simulated execution time, and storing them in the file you provided.