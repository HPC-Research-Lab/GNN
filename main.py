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
from multiprocessing import Value 


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
parser.add_argument('--epoch_num', type=int, default= 20,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=4,
                    help='Number of Pool')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='size of output node in a batch')
parser.add_argument('--orders', type=str, default='1,1,0',
                    help='Layer orders')
parser.add_argument('--samp_num', type=int, default=8192,
                    help='Number of sampled nodes per layer')
parser.add_argument('--cuda', type=str, default='0',
                    help='Avaiable GPU ID')
parser.add_argument('--sigmoid_loss', type=bool, default=True)
parser.add_argument('--global_permutation', type=bool, default=True)
parser.add_argument('--buffer_size', type=int, default=250000,
                    help='Number of buffered nodes on GPU')
parser.add_argument('--scale_factor', type=float, default=1,
                    help='Scale factor for skewed sampling')
parser.add_argument('--test', action='store_true')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--sampler', type=str, default='ladies')


args = parser.parse_args()


def init_process(rank, devices, world_size, fn, graph_data, buffer, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #os.environ['NCCL_DEBUG'] = 'INFO'
    #os.environ['NCCL_DEBUG_SUBSYS'] = 'GRAPH'
    #os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    if world_size > 1:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank + 1}/{world_size} process initialized.", flush=True)
    fn(rank, devices, world_size, graph_data, buffer)


def train(rank, devices, world_size, graph_data, buffer):

    torch.manual_seed(1234)

    device = devices[rank]
    torch.cuda.set_device(device)

    pool = ThreadPoolExecutor(max_workers=args.pool_num) 
        
    lap_matrix_tmp, labels_full_tmp, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data

    lap_matrix = sp.csr_matrix(lap_matrix_tmp[3])

    lap_matrix.indptr = np.frombuffer(lap_matrix_tmp[0].get_obj(), dtype=np.long)
    lap_matrix.indices = np.frombuffer(lap_matrix_tmp[1].get_obj(), dtype=np.long)
    lap_matrix.data = np.frombuffer(lap_matrix_tmp[2].get_obj(), dtype=np.float32)

    labels_full = sp.csr_matrix(labels_full_tmp[3])
    labels_full.indptr = np.frombuffer(labels_full_tmp[0].get_obj(), dtype=np.long)
    labels_full.indices = np.frombuffer(labels_full_tmp[1].get_obj(), dtype=np.long)
    labels_full.data = np.frombuffer(labels_full_tmp[2].get_obj(), dtype=np.int32)

    device_id_of_nodes, idx_of_nodes_on_device, gpu_buffers = buffer


    orders = args.orders.split(',')
    orders = [int(t) for t in orders]
    samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

    if args.sampler == 'subgraph':
        sampler = subgraph_sampler
    elif args.sampler == 'ladies':
        sampler = ladies_sampler
    else:
        sys.exit('sampler configuration is wrong')


    if rank == 0:
        print(args, flush=True)
        print('num batch per epoch: ', len(train_nodes) // args.batch_size, flush=True)
        #print(len(train_nodes))


    for oiter in range(1):
        if args.model == 'graphsage':
            encoder = GraphSage(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)
        elif args.model == 'gcn':
            encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)

        susage  = GNN(encoder = encoder, num_classes=num_classes, dropout=0.1, inp = feat_data.shape[1])
        print('num parameters: ', count_parameters(susage))
        susage.to(device)

        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=0.01)
        best_val = -1
        execution_time = 0.0
        data_movement_time = 0.0
        communication_time = 0.0

        
        iter = 0

        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []

            train_data = prepare_data(pool, lambda p: sampler(*p), train_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, args.batch_size, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, device, devices,  args.scale_factor, args.global_permutation, 'train')

            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e4 = torch.cuda.Event(enable_timing=True)


            for adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes in train_data:

                iter += 1

                optimizer.zero_grad()
                susage.train()

                for d in devices:
                    torch.cuda.synchronize(d)
                e1.record()

                input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])

                for i in range(world_size):
                    input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device)
                input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device)

                e2.record()
                for d in devices:
                    torch.cuda.synchronize(d)
                data_movement_time += e1.elapsed_time(e2)

                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                loss_train = loss(output, out_label, args.sigmoid_loss, device)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 5)


                if world_size > 1 and iter % 8 == 0:
                    average_model(susage, world_size)
                
                optimizer.step()

                e3.record()
                for d in devices:
                    torch.cuda.synchronize(d)
                execution_time += e1.elapsed_time(e3)

                train_losses += [loss_train.detach().tolist()]
                del loss_train

        
            if rank == 0:
                susage.eval()
                val_data = prepare_data(pool, lambda p: sampler(*p), valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, args.batch_size, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, device, devices,  mode='val')

                for adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes in val_data:    

                    input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])

                    for i in range(world_size):
                        input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device)
                    
                    input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device)

                    output = susage.forward(input_feat_data, adjs, sampled_nodes)
                    pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                    loss_valid = loss(output, out_label, args.sigmoid_loss, device).detach().tolist()
                    valid_f1, f1_mac = calc_f1(out_label.cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss)
                    print(("Epoch: %d (%.2fs)(%.2fs)(%.2fs)(%.2fs)(%.2fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, custom_sparse_ops.spmm_forward_time, custom_sparse_ops.spmm_backward_time, data_movement_time / 1000, communication_time / 1000, execution_time / 1000, np.average(train_losses), loss_valid, valid_f1), flush=True)
                    if valid_f1 > best_val + 1e-2:
                        best_val = valid_f1
                        torch.save(susage, './save/best_model.pt')
                    
                    

        if args.test == True and rank == 0:
            best_model = torch.load('./save/best_model.pt')
            best_model.eval()
            best_model.cpu()

            test_data = prepare_data(pool, lambda p: sampler(*p), test_nodes, samp_num_list, feat_data.shape[0], lap_matrix, labels_full, orders, 2048, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, device, devices, mode='test')

            correct = 0.0
            total = 0.0

            for adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, num_input_nodes, out_label, sampled_nodes in test_data:    
                input_feat_data = torch.cuda.FloatTensor(num_input_nodes, feat_data.shape[1])

                for i in range(world_size):
                    input_feat_data[input_nodes_mask_on_devices[i]] = gpu_buffers[i][nodes_idx_on_devices[i]].to(device)
                
                input_feat_data[input_nodes_mask_on_cpu] = feat_data[nodes_idx_on_cpu].to(device)
                    
                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                test_f1, f1_mac = calc_f1(out_label.cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss) 
                correct += test_f1 * out_label.shape[0]
                total += out_label.shape[0]

            print('Test f1 score: %.2f' % (correct / total), flush=True)
        



if __name__ == "__main__":

    devices = [int(i) for i in args.cuda.split(',')]
    print('gpu devices: ', devices, flush=True)
    world_size = len(devices)
    processes = []
    torch.multiprocessing.set_start_method('spawn')

    if 'ogbn' in args.dataset:
        graph_data = load_ogbn_data(args.dataset, os.environ['GNN_DATA_DIR'])
    else:
        graph_data = load_graphsaint_data(args.dataset, os.environ['GNN_DATA_DIR'])

    if args.model == 'graphsage':
        lap_matrix = row_normalize(graph_data[0])
    elif args.model == 'gcn':
        lap_matrix = row_normalize(graph_data[0] + sp.eye(graph_data[0].shape[0]))

  
    orders = args.orders.split(',')
    orders = [int(t) for t in orders]

    # buffer: device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers
    device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers = create_buffer(lap_matrix, graph_data, args.buffer_size, devices, args.dataset, sum(orders),  alpha=args.alpha)

    graph_data = create_shared_input_object(lap_matrix, graph_data)

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, devices, world_size, train, graph_data, (device_id_of_nodes_group[rank], idx_of_nodes_on_device_group[rank], gpu_buffers)))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

 
