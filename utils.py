import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import scipy.io as sio
import networkx as nx
from collections import defaultdict
import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from networkx.readwrite import json_graph
import json
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import torch.distributed as dist
import yaml


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sym_normalize(mx):
    """Sym-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1/2).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    
    mx = r_mat_inv.dot(mx).dot(c_mat_inv)
    return mx

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def generate_random_graph(n, e, prob = 0.1):
    idx = np.random.randint(2)
    g = nx.powerlaw_cluster_graph(n, e, prob) 
    adj_lists = defaultdict(set)
    num_feats = 8
    degrees = np.zeros(len(g), dtype=np.int64)
    edges = []
    for s in g:
        for t in g[s]:
            edges += [[s, t]]
            degrees[s] += 1
            degrees[t] += 1
    edges = np.array(edges)
    return degrees, edges, g, None 


def norm(l):
    return (l - np.average(l)) / np.std(l)

def stat(l):
    return np.average(l), np.sqrt(np.var(l))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
def get_laplacian(adj):
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

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



def package_mxl(mxl, device):
    res = []
    for mx in mxl:
        if mx != None:
            row, col, value = mx[0].to(device), mx[1].to(device), mx[2].to(device)
            sorted_idx = torch.argsort(row)
            row = row[sorted_idx]
            col = col[sorted_idx]
            value = value[sorted_idx]
            res.append((row, col, value, mx[3], mx[4]))
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


def average_grad(model):
        sendbuf = torch.cat(tuple((param.grad.data).view(param.grad.data.numel()) for i, param in enumerate(model.parameters()) if param.grad != None), 0)
        dist.all_reduce(sendbuf)

        start = 0
        for param in model.parameters():
            if param.grad != None:
                param.grad.data = sendbuf[start:start+param.grad.data.numel()].view(param.grad.data.size())
                start += param.grad.data.numel()

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f)
    return data