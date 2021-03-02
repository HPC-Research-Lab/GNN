from utils import *
import custom_sparse_ops

trans_time = 0.0

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros(n_out))
        self.scale = nn.Parameter(torch.ones(n_out))
        self.order = order
    def forward(self, x, adj, sampled_nodes):
        #global trans_time
        feat = x
        if self.order > 0:
            #profile(adj._indices())
            feat = custom_sparse_ops.spmm(*adj, feat)
            feat = torch.cat([x[sampled_nodes], feat], 1)
       # torch.cuda.synchronize()
       # t1 = time.time()
        out = F.elu(self.linear(feat))
       # torch.cuda.synchronize()
      #  trans_time += time.time() - t1
        mean = out.mean(dim=1).view(out.shape[0],1)
        var = out.var(dim=1, unbiased=False).view(out.shape[0], 1) + 1e-9
        return (out - mean) * self.scale * torch.rsqrt(var) + self.offset


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, orders, dropout):
        super(GCN, self).__init__()
        layers = len(orders)
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution((1+orders[0])*nfeat,  nhid, orders[0]))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution((1+orders[i+1])*nhid,  nhid, orders[i+1]))
    def forward(self, x, adjs, sampled_nodes):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx]))
        return x

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs, sampled_nodes):
        x = self.encoder(feat, adjs, sampled_nodes)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x