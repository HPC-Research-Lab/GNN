from utils import *
import torch
import custom_sparse_ops
import main
trans_time = 0.0

total_nodes = 14000
nodes_of_every_p = 3000
nhid = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y = [torch.zeros((total_nodes,nhid)).to(device),torch.zeros((total_nodes,nhid)).to(device),torch.zeros((total_nodes,nhid)).to(device)]

class GraphSageConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, order_id, bias=True):
        super(GraphSageConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linearW = nn.Linear(n_in,  n_out)
        self.linearB = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros((1+order)*n_out))
        self.scale = nn.Parameter(torch.ones((1+order)*n_out))
        self.order = order
        self.order_id = order_id
    def forward(self, x, adj, sampled_nodes):
        if self.order > 0:
            feat = custom_sparse_ops.spmm(adj, x)
            feat = self.linearW(feat)
            feat = 0.9 * feat + 0.1 * y[self.order_id][sampled_nodes,:]
            y[self.order_id][sampled_nodes,:] = feat.detach()
            unsampled_nodes = list(set(range(total_nodes))-set(sampled_nodes))
            y[self.order_id][unsampled_nodes,:] = 0.1 * y[self.order_id][unsampled_nodes,:]

            feat = torch.cat([self.linearB(x[sampled_nodes]), feat], 1)

        else:
            feat = self.linearW(x)
        out = F.elu(feat)
        mean = out.mean(dim=1).view(out.shape[0],1)
        var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
        return (out - mean) * self.scale * torch.rsqrt(var) + self.offset 

class GraphSage(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout):
        super(GraphSage, self).__init__()
        layers = len(orders)
        self.nhid = (1 + orders[-1]) * nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat,  nhid, orders[0], 0))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphSageConvolution((1+orders[i])*nhid,  nhid, orders[i+1], i+1))
    def forward(self, x, adjs, sampled_nodes):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx]))
        return x



class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, order_id, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros(n_out))
        self.scale = nn.Parameter(torch.ones(n_out))
        self.order = order
        self.order_id = order_id

    def forward(self, x, adj, sampled_nodes):
        feat = x
        if self.order > 0:
            feat = custom_sparse_ops.spmm(adj, feat)
            feat = self.linear(feat)
            feat = 0.9 * feat + 0.1 * y[self.order_id][sampled_nodes,:]
            y[self.order_id][sampled_nodes,:] = feat.detach()
            unsampled_nodes = list(set(range(nodes_of_every_p))-set(sampled_nodes))
            y[self.order_id][unsampled_nodes,:] = 0.1 * y[self.order_id][unsampled_nodes,:]
        else:
            feat = self.linear(feat)

        out = F.elu(feat)
        mean = out.mean(dim=1).view(out.shape[0],1)
        var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
        return (out - mean) * self.scale * torch.rsqrt(var) + self.offset


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout):
        super(GCN, self).__init__()
        layers = len(orders)
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid, orders[0], 0))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid, nhid, orders[i+1], i+1))
    def forward(self, x, adjs, sampled_nodes):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx]))
        return x

class GNN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs, sampled_nodes):
        x = self.encoder(feat, adjs, sampled_nodes)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x