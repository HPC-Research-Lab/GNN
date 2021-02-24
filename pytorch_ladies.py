#!/usr/bin/env python
# coding: utf-8


from utils import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp
from sklearn import metrics
import custom_sparse_ops
from concurrent.futures import ThreadPoolExecutor, as_completed



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
parser.add_argument('--queue_size', type=int, default= 32,
                    help='Max number of samples in the queue')
parser.add_argument('--batch_size', type=int, default=2048,
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

args = parser.parse_args()


def profile(idx):
    r = idx[0].cpu()
    c = idx[1].cpu()
   # print(r)
   # print(c)
    count = {}
    for i in range(len(r)):
        if r[i].item() not in count:
            count[r[i].item()] = 0
        count[r[i].item()] += 1
    print('r: ', sorted(count.values(), reverse=True)[0])

    count2 = {}
    for i in range(len(c)):
        if c[i].item() not in count2:
            count2[c[i].item()] = 0
        count2[c[i].item()] += 1
    print('c: ', sorted(count2.values(), reverse=True)[0])


class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros(n_out))
        self.scale = nn.Parameter(torch.ones(n_out))
        self.order = order
    def forward(self, x, adj):
        feat = x
        if self.order > 0:
            #profile(adj._indices())
            feat = custom_sparse_ops.spmm(adj, feat)
        out = F.elu(self.linear(feat))
        mean = out.mean(dim=1).view(out.shape[0],1)
        var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
        return (out - mean) * self.scale * torch.rsqrt(var) + self.offset


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout):
        super(GCN, self).__init__()
        layers = len(orders)
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid, orders[0]))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid, orders[i+1]))
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x



def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    #t1 = time.time()
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    orders1 = orders[::-1]
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(len(orders1)):
        if (orders1[d] == 0):
            adjs.append(None)
            continue
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = scipy.sparse.linalg.norm(U, ord=0, axis=0)
        #pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        #after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/np.clip(s_num * p[after_nodes], 1e-10, 1))
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    #if len(sampling_time) == 1:
     #   sampling_time[0] += time.time() - t1
    return adjs, previous_nodes, batch_nodes

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx if i>0 else None for i in orders], np.arange(num_nodes), batch_nodes

def prepare_data(pool, sampler, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, orders, mode='train', sampling_time = []):
    if mode == 'train':
        # sample p batches for training
        idxs = torch.randperm(len(train_nodes))
        num_batches = len(train_nodes) // args.batch_size
        if (len(train_nodes) % args.batch_size):
            num_batches += 1
        
        for i in range(0, num_batches, args.queue_size):
            samples = pool.map(sampler, [(np.random.randint(2**32 - 1), train_nodes[idxs[j*args.batch_size: min((j+1)*args.batch_size, len(idxs))]], samp_num_list, num_nodes, lap_matrix, orders) for j in range(i, min(i+args.queue_size, num_batches))])
            yield from samples
    elif mode == 'val':
        # sample a batch with more neighbors for validation
        idx = torch.randperm(len(valid_nodes))[:args.batch_size]
        batch_nodes = valid_nodes[idx]
        yield sampler((np.random.randint(2**32 - 1), batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, orders))

def package_mxl(mxl, device):
    res = []
    for mx in mxl:
        if mx != None:
            res.append(torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device).coalesce())
        else:
            res.append(None)
    return res

def loss(preds, labels, sigmoid_loss, device):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        norm_loss = torch.ones(preds.shape[0]).to(device)
        norm_loss /= preds.shape[0]
        if sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss*_ls).sum()

def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")



if __name__ == "__main__":
    if args.cuda != -1:
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")

    pool = ThreadPoolExecutor(max_workers=args.pool_num) 
        
        
    print(args.dataset, args.sample_method)
    adj_matrix, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)


    lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
    feat_data = torch.FloatTensor(feat_data).to(device)
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

        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()), lr=0.01)
        best_val = 0
        best_tst = -1
        cnt = 0
        execution_time = 0.0
        back_time = 0.0
        sampling_time = [0.0]
        #data_transfer_time = 0.0
        print('-' * 10)
        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []

            train_data = prepare_data(pool, lambda p: sampler(*p), train_nodes, valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, 'train')
            for adjs, input_nodes, output_nodes in train_data:    
                #t0 = time.time()
                adjs = package_mxl(adjs, device)
                #data_transfer_time += time.time() -  t0
                optimizer.zero_grad()
                torch.cuda.synchronize()
                t1 = time.time()
                susage.train()
                output = susage.forward(feat_data[input_nodes], adjs)
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
            
            
            susage.eval()
            val_data = prepare_data(pool, lambda p: sampler(*p), train_nodes, valid_nodes, samp_num_list, feat_data.shape[0], lap_matrix, orders, mode='val')

            for adjs, input_nodes, output_nodes in val_data:
                adjs = package_mxl(adjs, device)
                output = susage.forward(feat_data[input_nodes], adjs)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
                loss_valid = loss(output, labels_full[output_nodes], args.sigmoid_loss, device).detach().tolist()
                valid_f1, f1_mac = calc_f1(labels_full[output_nodes].detach().cpu().numpy(), pred.detach().cpu().numpy(), args.sigmoid_loss)
                print(("Epoch: %d (%.2fs)(%.2fs)(%.2fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") %                   (epoch, custom_sparse_ops.spmm_forward_time, custom_sparse_ops.spmm_backward_time, execution_time, np.average(train_losses), loss_valid, valid_f1))
                if valid_f1 > best_val + 1e-2:
                    best_val = valid_f1
                    torch.save(susage, './save/best_model.pt')

        best_model = torch.load('./save/best_model.pt')
        best_model.eval()
        best_model.cpu()
        '''
        If using batch sampling for inference:
        '''
        #     for b in np.arange(len(test_nodes) // args.batch_size):
        #         batch_nodes = test_nodes[b * args.batch_size : (b+1) * args.batch_size]
        #         adjs, input_nodes, output_nodes = sampler(np.random.randint(2**32 - 1), batch_nodes,
        #                                     samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
        #         adjs = package_mxl(adjs, device)
        #         output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
        #         test_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        #         test_f1s += [test_f1]
        
        '''
        If using full-batch inference:
        '''
        adjs, input_nodes, output_nodes = default_sampler(None, valid_nodes,
                                        None, len(feat_data), lap_matrix, orders)
        adjs = package_mxl(adjs, 'cpu')
        output = best_model.forward(feat_data[input_nodes].to('cpu'), adjs)[output_nodes]
        pred = nn.Sigmoid()(output) if args.sigmoid_loss else F.softmax(output, dim=1)
        test_f1, f1_mac = calc_f1(labels_full[output_nodes].cpu().numpy(), pred.detach().numpy(), args.sigmoid_loss)
        
        print('Test F1: %.3f' % (test_f1))

