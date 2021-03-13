## Overview

Transductive learning on graphs. Support: 
1) GraphSage and GCN models
2) Layer-wise and subgraph sampling
3) Multi-GPU training

## Datasets
Datasets are adopted from GraphSAINT. 

## Usage

To test with OGBN graphs:

```bash
python main.py --cuda='0' --dataset='ogbn-arxiv' --epoch_num 10
```

To test with graphsaint data, change the `load_ogbn_data` in 'main.py' to `load_graphsaint_data`, and then run the following:

```bash
python main.py --cuda='0' --dataset data/reddit --epoch_num 10
```

The default GNN model is GraphSage, to change the model to GCN add `--model='gcn'`in the command line.

The default sampler is layer-wise, to change the sampler to subgraph add `--sampler='subgraph'` in the command line.

To train with multiple GPUs, use `--cuda='0,1,2,3'`. 



