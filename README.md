## Overview

Transductive GCN adapted form LADIES. Main changes: 
1) add batch norm in each layer
2) add non-convolutional layers
3) sampling based on outgoing degree of the sampled nodes from previous layer

## Datasets
Datasets are adopted from GraphSAINT. 

## Usage

Execute the following scripts to train and evaluate the model:

```bash
python pytorch_ladies.py --cuda 0 --dataset data/ppi --epoch_num 1000 # Train GCN with LADIES on ppi graph.
```

