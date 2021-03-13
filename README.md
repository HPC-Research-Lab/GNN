## Overview

Transductive GCN adapted form LADIES. Main changes: 
1) add batch norm in each layer
2) add non-convolutional layers
3) sampling based on outgoing degree of the sampled nodes from previous layer

## Datasets
Datasets are adopted from GraphSAINT. 

## Usage

To test with ogbn graphs:

```bash
python main.py --cuda='0' --dataset='ogbn-arxiv' --epoch_num 10 --buffer_size 250000 
```

To test with graphsaint data, change the 'load_ogbn_data' in 'main.py' to 'load_graphsaint_data', and then run the following:

```bash
python main.py --cuda='0' --dataset data/reddit --epoch_num 10 --buffer_size 250000 
```
The default GNN model is GraphSage, to change the model to GCN add `--model='gcn'`in the command line.


