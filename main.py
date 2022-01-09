#!/usr/bin/env python
# coding: utf-8


from utils import *
from tqdm import tqdm
import argparse
import scipy
import torch.multiprocessing as mp
from models import *
from sampler import *
from preprocess import *
import torch.distributed as dist
import os
import subprocess
import threading


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='data/ppi',
                    help='Dataset name: ppi/reddit/amazon')
parser.add_argument('--model', type=str, default='graphsage',
                    help='GNN model: graphsage/gcn')
parser.add_argument('--nhid', type=int, default=512,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=4,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=4,
                    help='Number of Pool')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='size of output node in a batch')
parser.add_argument('--orders', type=str, default='1,1,1',
                    help='Layer orders')
parser.add_argument('--samp_num', type=int, default=8192,
                    help='Number of sampled nodes per layer')
parser.add_argument('--cuda', type=str, default='0',
                    help='Avaiable GPU ID')
parser.add_argument('--sigmoid_loss', type=bool, default=True)
parser.add_argument('--local_shuffle', action='store_true')
parser.add_argument('--buffer_size', type=float, default=0.2,
                    help='Ratio of nodes to buffer on GPU')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('--test', action='store_true')
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--sampler', type=str, default='ladies')
parser.add_argument('--pagraph', action='store_true')
parser.add_argument('--naive', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--locality_sampling', action='store_true')
parser.add_argument('--checkpoint', action='store_true')


args = parser.parse_args()


def train(rank, devices, world_size):

    global lap_matrix, labels_full, feat_data, num_classes, train_nodes, valid_nodes, test_nodes, device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers, gradients, barrier, sample_nodes_group, orders, scale_factor

    print(f"Rank {rank + 1}/{world_size} thread initialized.", flush=True)

    device = devices[rank]
    torch.cuda.set_device(device)

    pool = ThreadPoolExecutor(max_workers=args.pool_num) 
        
    device_id_of_nodes = device_id_of_nodes_group[rank]
    idx_of_nodes_on_device = idx_of_nodes_on_device_group[rank]

    samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

    if args.sampler == 'subgraph':
        sampler = subgraph_sampler
    elif args.sampler == 'ladies':
        sampler = ladies_sampler
    else:
        sys.exit('sampler configuration is wrong')

    if args.model == 'graphsage':
        encoder = GraphSage(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)
    elif args.model == 'gcn':
        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)
    if args.checkpoint == True:
        checkpoint = torch.load('./save/best_model.2x4GPU.locality_sampling.pt')
        susage = checkpoint['model']
        epoch_before = checkpoint['epoch']
        execution_time = checkpoint['exe_time']
        data_movement_time = checkpoint['move_time']
    else:
        susage  = GNN(encoder = encoder, num_classes=num_classes, dropout=0.1, inp = feat_data.shape[1])
        execution_time = 0.0
        data_movement_time = 0.0
    
    susage.to(device) 
    clk = time.CLOCK_THREAD_CPUTIME_ID


    optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=args.lr)
    best_val = -1
    communication_time = 0.0
    iter = 0
    factor_increase = args.locality_sampling
    factor_before = 0.0
    factor_after = 0.0

    for epoch in np.arange(args.epoch_num):
        if args.checkpoint == True:
            epoch += epoch_before
        susage.train()
        train_losses = []
        train_data = prepare_data(pool, sampler, train_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, args.batch_size, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, sample_nodes_group, device, devices, scale_factor, args.local_shuffle, 'train')


        for fut in as_completed(train_data):
            adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes = fut.result()

            iter += 1
            optimizer.zero_grad()
            susage.train()


            torch.cuda.synchronize(device)
            t1 = time.clock_gettime(clk)

            input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])
            for i in range(world_size):
                input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device).float()
            input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device, non_blocking=True).float()

            torch.cuda.synchronize(device)
            data_movement_time += time.clock_gettime(clk) - t1
    
            output = susage.forward(input_feat_data, adjs, sampled_nodes)


            loss_train = loss(output, out_label, args.sigmoid_loss, device)

            loss_train.backward()

            torch.nn.utils.clip_grad_norm_(susage.parameters(), 5)


            if world_size > 1:
                torch.cuda.synchronize(device)
                t2 = time.clock_gettime(clk)
                
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):

                    grad = torch.cat(tuple((param.grad.data).view(param.grad.data.numel()) for i, param in enumerate(susage.parameters()) if param.grad != None), 0)
                    gradients[rank] = grad
                    barrier.wait()
                    grad = sum([g if g.device == device else g.to(device) for g in gradients])
                
                    start = 0
                    for param in susage.parameters():
                        if param.grad != None:
                            param.grad.data = grad[start:start+param.grad.data.numel()].view(param.grad.data.size())
                            start += param.grad.data.numel()

                torch.cuda.synchronize(device)
                communication_time += time.clock_gettime(clk) - t2

            optimizer.step()

            torch.cuda.synchronize(device)
            execution_time += time.clock_gettime(clk) - t1

            train_losses += [loss_train.detach().tolist()]
            del loss_train
  
        if rank == 0:
            susage.eval()
            val_data = prepare_data(pool, sampler, valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, 128, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, sample_nodes_group, device, devices, scale_factor, args.local_shuffle, mode='val') 
            # this 'for' line would come to error
            for fut in as_completed(val_data):
                
                adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes = fut.result()
                input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])

                for i in range(world_size):
                    input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device).float()
                
                input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device, non_blocking=True).float()

                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                loss_valid = loss(output, out_label, args.sigmoid_loss, device).detach().tolist()
                valid_f1, f1_mac = calc_f1(out_label.cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss)
                print(("Epoch: %d (%.2fs)(%.2fs)(%.2fs)(%.2fs)(%.2fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f    scale_factor: %.3f     ratio: %3f") % (epoch, custom_sparse_ops.spmm_forward_time, custom_sparse_ops.spmm_backward_time, data_movement_time, communication_time, execution_time, np.average(train_losses), loss_valid, valid_f1, scale_factor, data_movement_time / execution_time), flush=True)
                if valid_f1 > best_val + 1e-2:
                    best_val = valid_f1
                    if args.locality_sampling == True:
                        torch.save({'model':susage, 'epoch':epoch, 'move_time':data_movement_time,'exe_time':execution_time}, './save/best_model.2x4GPU.locality_sampling.pt')
                    else:
                        torch.save({'model':susage, 'epoch':epoch, 'move_time':data_movement_time,'exe_time':execution_time}, './save/best_model.2x4GPU.ours.pt')

            if factor_increase == True:
                if scale_factor >=8:
                    factor_increase = False
                elif data_movement_time / execution_time >= 0.2:
                    factor_before = scale_factor
                    scale_factor *= 2
                elif data_movement_time / execution_time < 0.15 and scale_factor != 1:
                    factor_after = scale_factor
                    scale_factor = (factor_before + factor_after) / 2
                else:
                    factor_increase = False
                 
        
        barrier.wait()


    if args.test == True and rank == 0:
        best_model = torch.load('./save/best_model.pt')
        best_model.eval()
        best_model.cpu()
        test_data = prepare_data(pool, sampler, test_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, 128, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, sample_nodes_group, device, devices, scale_factor, args.local_shuffle, mode='test')

        correct = 0.0
        total = 0.0

        for fut in as_completed(test_data):    
            adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes = fut.result()
            input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])

            for i in range(world_size):
                input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device).float()
            
            input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device, non_blocking=True).float()
                
            output = susage.forward(input_feat_data, adjs, sampled_nodes)
            pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
            test_f1, f1_mac = calc_f1(out_label.cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss) 
            correct += test_f1 * out_label.shape[0]
            total += out_label.shape[0]

        print('Test f1 score: %.3f' % (correct / total), flush=True)
    



if __name__ == "__main__":

    print(args, flush=True)

    devices = [int(i) for i in args.cuda.split(',')]
    world_size = len(devices)
    orders = args.orders.split(',')
    orders = [int(t) for t in orders] 


    scale_factor = 1.0

    gradients = [None] * world_size

    barrier = threading.Barrier(world_size)

    if 'ogbn' in args.dataset or 'mag240m' in args.dataset:
        graph_data = load_ogbn_data(args.dataset, os.environ['GNN_DATA_DIR'])
    else:
        graph_data = load_graphsaint_data(args.dataset, os.environ['GNN_DATA_DIR'])

    if args.model == 'graphsage':
        lap_matrix = row_normalize(graph_data[0])
    elif args.model == 'gcn':
        lap_matrix = row_normalize(graph_data[0] + sp.eye(graph_data[0].shape[0]))

    _, labels_full, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data


    buffer_size = int(args.buffer_size * lap_matrix.shape[0])
    print('buffer_size: ', buffer_size)

    device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers, gpu_buffer_group, nodes_set_list = create_buffer(lap_matrix, graph_data, buffer_size, devices, args.dataset, sum(orders), alpha=args.alpha, pagraph_partition=args.pagraph, naive_partition=args.naive, random_partition=args.random)

    if args.local_shuffle == True and args.pagraph == True:
        assert(nodes_set_list != None)
        train_nodes = np.concatenate(nodes_set_list)
       

    sample_nodes_group = get_skewed_sampled_nodes(graph_data[0], gpu_buffer_group, orders)

    threads = []

    for rank in range(world_size):
        p = threading.Thread(target=train, args=(rank, devices, world_size))
        p.start()
        threads.append(p)
    
    for p in threads:
        p.join()

 
