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
    def __init__(self, n_in, n_out, order, idx, y=None, p=0, sco=False, mov_idx=None):
        super(GraphSageConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linearW = nn.Linear(n_in,  n_out)
        self.linearB = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros((1+order)*n_out))
        self.scale = nn.Parameter(torch.ones((1+order)*n_out))
        self.order = order
        self.sco = sco
        if (self.sco == True and self.training == True):
            self.y = y
            self.idx = idx
            self.p = p
            self.beta = 0.9
            self.mov_idx = mov_idx
    def forward(self, x, adj, sampled_nodes, nodes_per_layer, normfact_row, iterations, epoch):
        if self.sco == True and self.training == True:

            # epoch = 9 iteration = 2709
            # epoch = 19 iteration = 5719
            if epoch == 9:
                self.beta = 0.95
            if epoch == 19:
                self.beta = 0.99

            if self.order > 0:
                feat = custom_sparse_ops.spmm(adj, x)
                feat = self.beta * self.y[self.idx][nodes_per_layer] + feat - self.beta * feat.detach()
                self.y[self.idx] *= self.beta 
                self.y[self.idx][nodes_per_layer] = feat.detach()
                '''
                beta_vector = torch.zeros((len(nodes_per_layer),1)).cuda()
                beta_index = 0
                for i in self.mov_idx[self.idx][nodes_per_layer]:
                    beta_vector[beta_index] = self.beta ** (min(iterations,2709) - min(i,2709)) * 0.95 ** (max(min(iterations,5719) - max(i,2709),0)) * 0.99 ** (max(iterations,5719) - max(i,5719))
                    beta_index += 1

                feat = beta_vector * self.y[self.idx][nodes_per_layer] + feat - self.beta * feat.detach()
                self.y[self.idx][nodes_per_layer] = feat.detach()
                self.mov_idx[self.idx][nodes_per_layer] = iterations
                '''

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
    def __init__(self, nfeat, nhid, orders, dropout, y=None, p=0, iteration=0, sco=False, mov_idx=None):
        super(GraphSage, self).__init__()
        layers = len(orders)
        self.nhid = (1 + orders[-1]) * nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat,  nhid, orders[0], 0, y=y, p=p, sco=sco, mov_idx=mov_idx))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphSageConvolution((1+orders[i])*nhid,  nhid, orders[i+1], i+1, y=y, p=p, sco=sco, mov_idx=mov_idx))
    def forward(self, x, adjs, sampled_nodes, nodes_per_layer, normfact_row_list, iterations, epoch):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx], nodes_per_layer[idx], normfact_row_list, iterations, epoch))
        return x



class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, order, y=None, idx=-1, p=0, sco=False, mov_idx=None):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.offset = nn.Parameter(torch.zeros(n_out))
        self.scale = nn.Parameter(torch.ones(n_out))
        self.order = order
        self.sco = sco
        if (self.sco == True and self.training == True):
            self.y = y
            self.idx = idx
            self.p = p
            self.mov_idx = mov_idx
            self.beta = 0.9
            self.judge = True
            self.judge1 = True
    
    def forward(self, x, adj, sampled_nodes, nodes_per_layer, iterations, epoch):
        if (self.training == True and self.sco == True):
            feat = x

            # epoch = 9 iteration = 2709
            # epoch = 19 iteration = 5719
            if epoch == 9:
                self.beta = 0.95
            if epoch == 19:
                self.beta = 0.99

            if self.order > 0:
                feat = custom_sparse_ops.spmm(adj, feat)
                feat = self.beta * self.y[self.idx][nodes_per_layer] + feat - self.beta * feat.detach()
                self.y[self.idx] *= self.beta 
                self.y[self.idx][nodes_per_layer] = feat.detach()
                '''
                beta_vector = torch.zeros((len(nodes_per_layer),1)).cuda()
                beta_index = 0
                for i in self.mov_idx[self.idx][nodes_per_layer]:
                    beta_vector[beta_index] = self.beta ** (min(iterations,2709) - min(i,2709)) * 0.95 ** (max(min(iterations,5719) - max(i,2709),0)) * 0.99 ** (max(iterations,5719) - max(i,5719))
                    beta_index += 1

                feat = beta_vector * self.y[self.idx][nodes_per_layer] + feat - self.beta * feat.detach()
                self.y[self.idx][nodes_per_layer] = feat.detach()
                self.mov_idx[self.idx][nodes_per_layer] = iterations
                '''
    
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
    def __init__(self, nfeat, nhid, orders, dropout, y=None, p=0, iteration=0, sco=False, mov_idx=None):
        super(GCN, self).__init__()
        layers = len(orders)
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid, orders[0], y=y, idx=0, p=p, sco=sco, mov_idx=mov_idx))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid, nhid, orders[i+1], y=y, idx=i+1, p=p, sco=sco, mov_idx=mov_idx))
    def forward(self, x, adjs, sampled_nodes, nodes_per_layer, normfact_row_list, iterations, epoch):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx], sampled_nodes[idx] if sampled_nodes != None else None, nodes_per_layer[idx], iterations, epoch))
        return x

class GNN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(GNN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs, sampled_nodes, nodes_per_layer, normfact_row_list, iterations, epoch):
        x = self.encoder(feat, adjs, sampled_nodes, nodes_per_layer, normfact_row_list, iterations, epoch)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x