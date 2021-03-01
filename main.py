#!/usr/bin/env python
# coding: utf-8


from utils import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp
from models import *
from sampler import *
from preprocess import *


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
parser.add_argument('--pool_num', type=int, default= 16,
                    help='Number of Pool')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--orders', type=str, default='1,0,1,0',
                    help='Layer orders')
parser.add_argument('--samp_num', type=int, default=16384,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--sigmoid_loss', type=bool, default=True)
parser.add_argument('--buffer_size', type=int, default=10000,
                    help='Number of buffered nodes on GPU')

args = parser.parse_args()


if __name__ == "__main__":
    if args.cuda != -1:
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    pool = ThreadPoolExecutor(max_workers=args.pool_num) 
        
        
    print(args.dataset, args.sample_method)
    adj_matrix, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)

    lap_matrix = row_normalize(adj_matrix)
    feat_data = torch.FloatTensor(feat_data)
    print('sigmoid_loss: ', args.sigmoid_loss)
    print('batch_size: ', args.batch_size)
    print('num batch per epoch: ', len(train_nodes) // args.batch_size)
    if args.sigmoid_loss == True:
        labels_full = torch.from_numpy(class_arr).to(device)
    else:
        labels_full = torch.from_numpy(class_arr.argmax(axis=1).astype(np.int64)).to(device)

    orders = args.orders.split(',')
    orders = [int(t) for t in orders]




    if args.sample_method == 'ladies':
        sampler = ladies_sampler
    elif args.sample_method == 'fastgcn':
        sampler = fastgcn_sampler
    elif args.sample_method == 'full':
        sampler = default_sampler


    samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

    for oiter in range(1):
        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, orders=orders, dropout=0.1).to(device)
        susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.1, inp = feat_data.shape[1])
        susage.to(device)


        buffer, buffer_map, buffer_mask = create_buffer(np.array(np.sum(adj_matrix, axis=0))[0]
, feat_data, args.buffer_size, device)
        samples = np.zeros(adj_matrix.shape[1])

        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=0.01)
        best_val = 0
        best_tst = -1
        cnt = 0
        execution_time = 0.0
        back_time = 0.0
        print('-' * 10)
        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []

            train_data = prepare_data(pool, sampler, train_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, 'train')
            for fut in as_completed(train_data):    
                adjs, input_nodes, output_nodes, sampled_nodes = fut.result()
                adjs = package_mxl(adjs, device)
                #data_transfer_time += time.time() -  t0
                optimizer.zero_grad()
                torch.cuda.synchronize()
                t1 = time.time()
                susage.train()
                input_feat_data = torch.cuda.FloatTensor(len(input_nodes), feat_data.shape[1])
                input_nodes_idx = buffer_map[input_nodes]
                input_nodes_mask = buffer_mask[input_nodes]
                feat_gpu_idx = input_nodes_mask == True
                feat_cpu_idx = input_nodes_mask == False
                gpu_nodes_idx = input_nodes_idx[feat_gpu_idx]
                cpu_nodes_idx = input_nodes_idx[feat_cpu_idx]
                input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
                input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                loss_train = loss(output, labels_full[output_nodes], args.sigmoid_loss, device)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 5)
                optimizer.step()
                torch.cuda.synchronize()
                execution_time += time.time() - t1
                train_losses += [loss_train.detach().tolist()]
                del loss_train

                for sn in sampled_nodes:
                    samples[sn] += 1

           
            
            
            susage.eval()
            val_data = prepare_data(pool, sampler, valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, mode='val')

            for fut in as_completed(val_data):    
                adjs, input_nodes, output_nodes, sampled_nodes = fut.result()
                adjs = package_mxl(adjs, device)
                input_feat_data = torch.cuda.FloatTensor(len(input_nodes), feat_data.shape[1])
                input_nodes_idx = buffer_map[input_nodes]
                input_nodes_mask = buffer_mask[input_nodes]
                feat_gpu_idx = input_nodes_mask == True
                feat_cpu_idx = input_nodes_mask == False
                gpu_nodes_idx = input_nodes_idx[feat_gpu_idx]
                cpu_nodes_idx = input_nodes_idx[feat_cpu_idx]
                input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
                input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
                output = susage.forward(input_feat_data, adjs, sampled_nodes)
                #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                loss_valid = loss(output, labels_full[output_nodes], args.sigmoid_loss, device).detach().tolist()
                valid_f1, f1_mac = calc_f1(labels_full[output_nodes].detach().cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss)
                print(("Epoch: %d (%.2fs)(%.2fs)(%.2fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") %                   (epoch, custom_sparse_ops.spmm_forward_time, custom_sparse_ops.spmm_backward_time, execution_time, np.average(train_losses), loss_valid, valid_f1))
                if valid_f1 > best_val + 1e-2:
                    best_val = valid_f1
                    torch.save(susage, './save/best_model.pt')

            if (epoch+1) % 4 == 0:
                buffer, buffer_map, buffer_mask = create_buffer(samples, feat_data, args.buffer_size, device)

        best_model = torch.load('./save/best_model.pt')
        best_model.eval()
        best_model.cpu()
        '''
        If using batch sampling for inference:
        '''
        test_data = prepare_data(pool, sampler, test_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, args.batch_size, mode='test')

        correct = 0.0
        total = 0.0

        for fut in as_completed(test_data):    
            adjs, input_nodes, output_nodes, sampled_nodes = fut.result()
            adjs = package_mxl(adjs, device)
            input_feat_data = torch.cuda.FloatTensor(len(input_nodes), feat_data.shape[1])
            input_nodes_idx = buffer_map[input_nodes]
            input_nodes_mask = buffer_mask[input_nodes]
            feat_gpu_idx = input_nodes_mask == True
            feat_cpu_idx = input_nodes_mask == False
            gpu_nodes_idx = input_nodes_idx[feat_gpu_idx]
            cpu_nodes_idx = input_nodes_idx[feat_cpu_idx]
            input_feat_data[feat_gpu_idx] = buffer[gpu_nodes_idx]
            input_feat_data[feat_cpu_idx] = feat_data[cpu_nodes_idx].to(device)
            output = susage.forward(input_feat_data, adjs, sampled_nodes)
            #output = susage.forward(feat_data[input_nodes].to(device), adjs, sampled_nodes)
            if args.sample_method == 'full':
                output = output[output_nodes]
            pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
            test_f1, f1_mac = calc_f1(labels_full[output_nodes].cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss) 
            correct += test_f1 * len(output_nodes)
            total += len(output_nodes)


        print('Test f1 score: %.2f' % (correct / total))
        
