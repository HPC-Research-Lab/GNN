#!/bin/bash

export GNN_DATA_DIR=/data/not_backed_up/shared/gnn_data/
for bs in 0 320000 640000 960000 1280000 1600000; do
for a in 0 1 0.2 0.8 0.4 0.6; do
  	echo $bs $a; 
	python main.py --dataset amazon --cuda='0,1' --batch_size 2048 --samp_num 4096 --epoch_num 4 --buffer_size $bs --model gcn --sampler subgraph --orders='1,1,1' --alpha $a; 
	killall python;
	sleep 60;
done
done
