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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='data/ppi',
                    help='Dataset name: ppi/reddit')
parser.add_argument('--nhid', type=int, default=512,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 1000,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 4,
                    help='Number of Pool')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--orders', type=str, default='1,0,1,0',
                    help='Layer orders')
parser.add_argument('--samp_num', type=int, default=16384,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=str, default='0',
                    help='Avaiable GPU ID')
parser.add_argument('--sigmoid_loss', type=bool, default=True)
parser.add_argument('--buffer_size', type=int, default=10000,
                    help='Number of buffered nodes on GPU')
parser.add_argument('--scale_factor', type=float, default=1,
                    help='Scale factor for skewed sampling')
parser.add_argument('--update_buffer_period', type=int, default=0,
                    help='Period of GPU buffer being updated')
parser.add_argument('--global_permutation', dest='global_permutation', action='store_true')
parser.set_defaults(global_permutation=False)


args = parser.parse_args()


def init_process(rank, device_id, world_size, fn, train_data, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank + 1}/{world_size} process initialized.")
    fn(rank, device_id, world_size, train_data)


def train(rank, device_id, world_size, train_data):
    if args.cuda != -1:
        device = torch.device("cuda:" + str(device_id))
    else:
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    pool = ThreadPoolExecutor(max_workers=args.pool_num) 
        
        
    adj_matrix, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = train_data

    lap_matrix = row_normalize(adj_matrix)
    feat_data = torch.FloatTensor(feat_data)

    if args.sigmoid_loss == True:
        labels_full = torch.from_numpy(class_arr).to(device)
    else:
        labels_full = torch.from_numpy(class_arr.argmax(axis=1).astype(np.int64)).to(device)

    orders = args.orders.split(',')
    orders = [int(t) for t in orders]
    samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])
    sampler = ladies_sampler




    if rank == 0:
        print(args.dataset, args.sample_method)
        print('sigmoid_loss: ', args.sigmoid_loss)
        print('batch_size: ', args.batch_size)
        print('num batch per epoch: ', len(train_nodes) // args.batch_size)


    for oiter in range(1):
        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)
        susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.1, inp = feat_data.shape[1])
        susage.to(device)

        if args.update_buffer_period > 0:
            sp_mat = get_sample_matrix(adj_matrix, train_nodes, orders, rank, world_size)

        buffer, buffer_map, buffer_mask = create_buffer(np.array(np.sum(adj_matrix, axis=0))[0], feat_data, args.buffer_size, device)

        samples = np.zeros(adj_matrix.shape[1])

        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=0.01)
        best_val = 0
        best_tst = -1
        cnt = 0
        execution_time = 0.0
        back_time = 0.0
        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []

            train_data = prepare_data(pool, sampler, train_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, rank, world_size, buffer_map, buffer_mask, device, args.scale_factor, args.global_permutation, 'train')
            for fut in as_completed(train_data):
                adjs, gpu_nodes_idx, cpu_nodes_idx, feat_gpu_idx, feat_cpu_idx, output_nodes, sampled_nodes, sampled_cols = fut.result()

                optimizer.zero_grad()
                susage.train()

                torch.cuda.synchronize()
                t1 = time.time()
                input_feat_data = torch.cuda.FloatTensor(len(gpu_nodes_idx)+len(cpu_nodes_idx), feat_data.shape[1])
                input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
                input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
                loss_train = loss(output, labels_full[output_nodes], args.sigmoid_loss, device)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 5)

                # communication is expensive
                if world_size > 1:
                    for param in susage.parameters():
                        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                optimizer.step()

                train_losses += [loss_train.detach().tolist()]
                del loss_train

                if args.update_buffer_period > 0 and epoch < args.update_buffer_period:
                    sp_mat = update_sampled_matrix(output_nodes, sampled_cols, sp_mat)
                    
                torch.cuda.synchronize()
                execution_time += time.time() - t1
        
            if args.update_buffer_period > 0 and epoch == args.update_buffer_period:
                sp_prob = reorder_and_restart(adj_matrix, train_nodes, sp_mat, rank, world_size)
                buffer, buffer_map, buffer_mask = create_buffer(sp_prob, feat_data, args.buffer_size, device)

            if rank == 0:
                susage.eval()
                val_data = prepare_data(pool, sampler, valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, rank, world_size, buffer_map, buffer_mask, device, mode='val')

                for fut in as_completed(val_data):    
                    adjs, gpu_nodes_idx, cpu_nodes_idx, feat_gpu_idx, feat_cpu_idx, output_nodes, sampled_nodes, sampled_cols = fut.result()
                    #adjs = package_mxl(adjs, device)
                    input_feat_data = torch.cuda.FloatTensor(len(gpu_nodes_idx)+len(cpu_nodes_idx), feat_data.shape[1])
                    input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
                    input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
                    output = susage.forward(input_feat_data, adjs, sampled_nodes)
                    #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
                    pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                    loss_valid = loss(output, labels_full[output_nodes], args.sigmoid_loss, device).detach().tolist()
                    valid_f1, f1_mac = calc_f1(labels_full[output_nodes].detach().cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss)
                    print(("Epoch: %d (%.2fs)(%.2fs)(%.2fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") %                   (epoch, custom_sparse_ops.spmm_forward_time, custom_sparse_ops.spmm_backward_time, execution_time, np.average(train_losses), loss_valid, valid_f1))
                    if valid_f1 > best_val + 1e-2:
                        best_val = valid_f1
                        torch.save(susage, './save/best_model.pt')

    
                    
                    

        if rank == 0:
            best_model = torch.load('./save/best_model.pt')
            best_model.eval()
            best_model.cpu()
            '''
            If using batch sampling for inference:
            '''
            test_data = prepare_data(pool, sampler, test_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, rank, world_size, buffer_map, buffer_mask, device, mode='test')

            correct = 0.0
            total = 0.0

            for fut in as_completed(test_data):    
                adjs, gpu_nodes_idx, cpu_nodes_idx, feat_gpu_idx, feat_cpu_idx, output_nodes, sampled_nodes, sample_cols = fut.result()
                #adjs = package_mxl(adjs, device)
                input_feat_data = torch.cuda.FloatTensor(len(gpu_nodes_idx)+len(cpu_nodes_idx), feat_data.shape[1])
                input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
                input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
                pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                test_f1, f1_mac = calc_f1(labels_full[output_nodes].cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss) 
                correct += test_f1 * len(output_nodes)
                total += len(output_nodes)


            print('Test f1 score: %.2f' % (correct / total))
        



if __name__ == "__main__":

    devices = [int(i) for i in args.cuda.split(',')]
    print('gpu devices: ', devices)
    world_size = len(devices)
    processes = []
    torch.multiprocessing.set_start_method('spawn')

    train_data = load_data(args.dataset)

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, devices[rank], world_size, train, train_data))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
