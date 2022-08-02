# Graph Data Placement Implementation
**Rethinking graph data placement for graph neural network training on multiple GPUs**

Shihui Song, Peng Jiang

https://dl.acm.org/doi/pdf/10.1145/3524059.3532384

Abstract: *Graph partitioning is commonly used for dividing graph data for parallel processing. While they achieve good performance for the traditional graph processing algorithms, the existing graph partitioning methods are unsatisfactory for data-parallel GNN training on GPUs. In this work, we rethink the graph data placement problem for large-scale GNN training on multiple GPUs. We find that loading input features is a performance bottleneck for GNN training on large graphs that cannot be stored on GPU. To reduce the data loading overhead, we first propose a performance model of data movement among CPU and GPUs in GNN training. Then, based on the performance model, we provide an efficient algorithm to divide and distribute the graph data onto multiple GPUs so that the data loading time is minimized. For cases where data placement alone cannot achieve good performance, we propose a locality-aware neighbor sampling technique to further reduce the data movement overhead without losing accuracy. Our experiments with graphs of different sizes on different numbers of GPUs show that our techniques not only achieve smaller data loading time but also incur much less preprocessing overhead than the existing graph partitioning methods.*
## Requirements
* Python >= 3.8
* PyTorch >= 1.8.0
* OGB >= 1.3.3

## Training
Our implementation includes four branches.
To train models with **our** method on Reddit, run this command using **main branch**:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04
```
To train models with **our** method with **locality-aware neighbor sampling** on Reddit, we just need to add --locality_sampling:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04 --locality_sampling
```
To train models with **pagraph** partition on Reddit, we just need to add --local_shuffle --pagraph:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04 --local_shuffle --pagraph
```
To train models with **naive** partition on Reddit, we just need to add --naive:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04 --naive
```
To train models with **random** partition on Reddit, we just need to add --naive --random:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04 --naive --random
```
To train models with **metis** partition on Reddit, run this command under **metis branch**:
```
python main.py --dataset='reddit' --cuda='0,1,2,3' --alpha 0 --batch_size 512 --epoch_num 30 --buffer_size 0.1 --model='graphsage' --sampler='ladies' --samp_num 8192 --lr 0.04
```
To train models with **metis** partition on Reddit with **(2+2)GPU** configuration, run the same command under **pair_metis branch**.

To train models with **methods except metis** on Reddit with **(2+2)GPU** configuration, run the same commands under **pair branch**.

## Citation
If you use this library in a research paper, please cite this repository.
```
@inproceedings{DBLP:conf/ics/SongJ22,
  author    = {Shihui Song and Peng Jiang},
  title     = {Rethinking graph data placement for graph neural network training on multiple GPUs},
  booktitle = {{ICS} '22: 2022 International Conference on Supercomputing},
  pages     = {39:1--39:10},
  publisher = {{ACM}},
  year      = {2022},
}
```
