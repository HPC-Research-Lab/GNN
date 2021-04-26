from utils import *
import torch
import custom_sparse_ops
import main
trans_time = 0.0

#total_nodes = 14000
#nodes_of_every_p = 3000
#nhid = 512
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#y = [torch.zeros((total_nodes,nhid)).to(device),torch.zeros((total_nodes,nhid)).to(device),torch.zeros((total_nodes,nhid)).to(device)]

class GraphSageConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, idx, y=None, p=0, bias=True):
        super(GraphSageConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linearW = nn.Linear(n_in,  n_out)
        self.linearB = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros((1+order)*n_out))
        self.scale = nn.Parameter(torch.ones((1+order)*n_out))
        self.order = order
        if (self.training == True):
            self.y = y
            self.idx = idx
            self.p = p
            self.feat_id = 0
    def forward(self, x, adj, sampled_nodes, nodes_per_layer, iterations):
        if self.training == True:
            if self.order > 0:
                feat = custom_sparse_ops.spmm(adj, x)
                if self.idx > 0:
                    self.feat_id = self.y[1].shape[1]
                feat[:,self.feat_id:] = 0.9 * feat[:,self.feat_id:] + 0.1 * self.y[self.idx][nodes_per_layer,:]
                self.y[self.idx][nodes_per_layer,:] = feat[:,self.feat_id:].detach()
                index = torch.ones(self.y[0].shape[0], dtype=bool)
                index[nodes_per_layer] = False
                self.y[self.idx][index] *= 0.9 

                feat = torch.cat([self.linearB(x[sampled_nodes]), self.linearW(feat)], 1)
            else:
                feat = self.linearW(x)
            out = F.elu(feat)
            mean = out.mean(dim=1).view(out.shape[0],1)
            var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
            return (out - mean) * self.scale * torch.rsqrt(var) + self.offset 
        else:
            if self.order > 0:
                feat = custom_sparse_ops.spmm(adj, x)
                feat = torch.cat([self.linearB(x[sampled_nodes]), self.linearW(feat)], 1)
            else:
                feat = self.linearW(x)
            out = F.elu(feat)
            mean = out.mean(dim=1).view(out.shape[0],1)
            var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
            return (out - mean) * self.scale * torch.rsqrt(var) + self.offset 

class GraphSage(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout, y=None, p=0, iteration=0):
        super(GraphSage, self).__init__()
        layers = len(orders)
        self.nhid = (1 + orders[-1]) * nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat,  nhid, orders[0], 0, y=y, p=p))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphSageConvolution((1+orders[i])*nhid,  nhid, orders[i+1], i+1, y=y, p=p))
    def forward(self, x, adjs, sampled_nodes, nodes_per_layer, iterations):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx], nodes_per_layer[idx], iterations))
        return x



class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, y=None, idx=-1, p=0, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros(n_out))
        self.scale = nn.Parameter(torch.ones(n_out))
        self.order = order
        if (self.training == True):
            self.y = y
            self.idx = idx
            self.p = p

    def forward(self, x, adj, sampled_nodes, nodes_per_layer, iterations):
        if (self.training == True):
            feat = x
            if self.order > 0:
                feat = custom_sparse_ops.spmm(adj, feat)
                feat = 0.9 * self.y[self.idx][nodes_per_layer] + 0.1 * feat
                self.y[self.idx][nodes_per_layer] = feat.detach()
                index = torch.ones(self.y[0].shape[0], dtype=bool)
                index[nodes_per_layer] = False
                self.y[self.idx][index] *= 0.9 

            out = F.elu(self.linear(feat))
            mean = out.mean(dim=1).view(out.shape[0],1)
            var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
            return (out - mean) * self.scale * torch.rsqrt(var) + self.offset
        else:
            feat = x
            if self.order > 0:
                feat = torch.spmm(adj, feat)
            out = F.elu(self.linear(feat))
            mean = out.mean(dim=1).view(out.shape[0],1)
            var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
            return (out - mean) * self.scale * torch.rsqrt(var) + self.offset            


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout, y=None, p=0, iteration=0):
        super(GCN, self).__init__()
        layers = len(orders)
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid, orders[0], y=y, idx=0, p=p))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid, nhid, orders[i+1], y=y, idx=i+1, p=p))
    def forward(self, x, adjs, sampled_nodes, nodes_per_layer, iterations):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx] if sampled_nodes != None else None, nodes_per_layer[idx], iterations))
        return x

class GNN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs, sampled_nodes, nodes_per_layer, iterations):
        x = self.encoder(feat, adjs, sampled_nodes, nodes_per_layer, iterations)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x