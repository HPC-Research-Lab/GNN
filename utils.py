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
import scipy
from sklearn.preprocessing import StandardScaler



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(prefix):
    # adj_full: graph edges stored in coo format, role: dict storing indices of train, val, test nodes
    # feats: features of all nodes, class_map: label of all nodes
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.float)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = role['tr']
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1

    return adj_full, class_arr, feats, num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te'])

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

def get_sparse(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

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
    return indices, values, shape


def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
def get_laplacian(adj):
    adj = row_normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 