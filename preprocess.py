from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.cuda.memory import max_memory_cached
from utils import *
import torch.distributed as dist
import subprocess
from itertools import groupby 
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, dropout_adj
import multiprocessing as mp
import pickle
import matplotlib
import heapq
from random import shuffle


def load_graphsaint_data(graph_name, root_dir):
    # adj_full: graph edges stored in coo format, role: dict storing indices of train, val, test nodes
    # feats: features of all nodes, class_map: label of all nodes
    adj_full = sp.load_npz(f'{root_dir}/{graph_name}/adj_full.npz').astype(np.float32)
    role = json.load(open(f'{root_dir}/{graph_name}/role.json'))
    feats = np.load(f'{root_dir}/{graph_name}/feats.npy').astype(np.float32)
    class_map = json.load(open(f'{root_dir}/{graph_name}/class_map.json'))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = role['tr']
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = sp.lil_matrix((num_vertices, num_classes), dtype=np.int32)
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = sp.lil_matrix((num_vertices, num_classes), dtype=np.int32)
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k, v-offset] = 1

    class_arr = class_arr.tocsr()
    
    print('feat dim: ', feats.shape, flush=True)
    print('label dim: ', class_arr.shape, flush=True)
    

    return (adj_full, class_arr, torch.FloatTensor(feats), num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))


def load_ogbn_data(graph_name, root_dir):

    if graph_name == 'mag240m_kddcup2021':
        return load_mag240M(root_dir)

    dataset = PygNodePropPredDataset(graph_name, root=root_dir)
    split_idx = dataset.get_idx_split()
    data = dataset[0]


    row, col = data.edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    num_vertices = data.num_nodes

    adj_full = sp.csr_matrix(([1]*len(row), (row, col)), shape=(num_vertices, num_vertices), dtype=np.float32)
    row = None
    col = None

    feats = data.x
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    class_data = data.y.data.flatten()
    assert(len(class_data) == num_vertices)

    class_data_compact = class_data[~torch.isnan(class_data)]

    max_class_idx = torch.max(class_data_compact)
    min_class_idx = torch.min(class_data_compact)

    num_classes = max_class_idx - min_class_idx + 1
    num_classes = int(num_classes.item())

    class_arr = sp.lil_matrix((num_vertices, num_classes), dtype=np.int32)
    for i in range(len(class_data)):
        if not torch.isnan(class_data[i]): 
            class_arr[i, class_data[i]-min_class_idx] = 1
    
    class_arr = class_arr.tocsr()

    print('feat dim: ', feats.shape, flush=True)
    print('label dim: ', class_arr.shape, flush=True)

    return (adj_full, class_arr, feats, num_classes, train_idx, valid_idx, test_idx)


def load_mag240M(root_dir):
    from ogb.lsc import MAG240MDataset
    data = MAG240MDataset(root = root_dir)

    row, col = data.edge_index('paper', 'paper')
    row, col = np.concatenate([row, col]), np.concatenate([col, row])
    num_vertices = data.num_papers

    adj_full = sp.csr_matrix(([1]*len(row), (row, col)), shape=(num_vertices, num_vertices), dtype=np.float32)
    row = None
    col = None

    feats = torch.from_numpy(data.paper_feat)

    split_idx = data.get_idx_split()

    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    class_data = data.paper_label

    max_class_idx = None
    min_class_idx = None

    for c in class_data:
        if not np.isnan(c):
            if max_class_idx == None or max_class_idx < c:
                max_class_idx = c
            if min_class_idx == None or min_class_idx > c:
                min_class_idx = c

    num_classes = max_class_idx - min_class_idx + 1
    num_classes = int(num_classes.item())

    class_arr = sp.lil_matrix((num_vertices, num_classes), dtype=np.int32)
    for i in range(len(class_data)):
        if not np.isnan(class_data[i]): 
            class_arr[i, class_data[i]-min_class_idx] = 1
    
    class_arr = class_arr.tocsr()

    print('feat dim: ', feats.shape, flush=True)
    print('label dim: ', class_arr.shape, flush=True)

    return (adj_full, class_arr, feats, num_classes, train_idx, valid_idx, test_idx)


    
def reorder_graphsaint_graph(adj_full, adj_train, feats, class_map, role):
    pv_list = np.array(
        [
            adj_train.data[
                adj_train.indptr[v] : adj_train.indptr[v + 1]
            ].sum()
            for v in range(adj_train.shape[0])
        ],
        dtype=np.int64,
    )
    rate_nodes = sorted(range(len(pv_list)), key=lambda k: pv_list[k], reverse=True)

    rate_nodes_dict = {}
    for i in range(len(rate_nodes)):
        rate_nodes_dict[rate_nodes[i]] = i

    adj_full_indices = []
    adj_full_indptr = []
    adj_train_indices = []
    adj_train_indptr = []
    adj_full_indptr.append(0)
    adj_train_indptr.append(0)
    index_full = 0
    index_train = 0
    for v in rate_nodes:
        v_s_full = adj_full.indptr[v]
        v_e_full = adj_full.indptr[v + 1]
        for col in adj_full.indices[v_s_full:v_e_full]:
            adj_full_indices.append(rate_nodes_dict[col])
        data_col = adj_full_indices[v_s_full:v_e_full]
        index_full += v_e_full - v_s_full
        adj_full_indptr.append(index_full)

        v_s_train = adj_train.indptr[v]
        v_e_train = adj_train.indptr[v + 1]
        for col in adj_train.indices[v_s_train:v_e_train]:
            adj_train_indices.append(rate_nodes_dict[col])        
        index_train += v_e_train - v_s_train
        adj_train_indptr.append(index_train)     
    adj_full.indices = np.array(adj_full_indices, dtype=np.int32)
    adj_full.indptr = np.array(adj_full_indptr, dtype=np.int32)
    adj_train.indices = np.array(adj_train_indices, dtype=np.int32)
    adj_train.indptr = np.array(adj_train_indptr, dtype=np.int32)
    
    feats_reorder = feats[rate_nodes]
    class_map_reorder = {}
    role_reorder = {}
    for i in range(adj_train.shape[0]):
        class_map_reorder[i] = class_map[rate_nodes[i]]
    role_reorder['tr'] = []
    role_reorder['va'] = []
    role_reorder['te'] = []
    for i in role['tr']:
        role_reorder['tr'].append(rate_nodes_dict[i])
    for i in role['va']:
        role_reorder['va'].append(rate_nodes_dict[i])
    for i in role['te']:
        role_reorder['te'].append(rate_nodes_dict[i])

    return adj_full, adj_train, feats_reorder, class_map_reorder, role_reorder


def reorder_ogbn_graph(adj_full, feats, class_data, train_idx, valid_idx, test_idx):
    pv_list = np.array(
        [
            adj_full.data[
                adj_full.indptr[v] : adj_full.indptr[v + 1]
            ].sum()
            for v in range(adj_full.shape[0])
        ],
        dtype=np.int64,
    )

    rate_nodes = sorted(range(len(pv_list)), key=lambda k: pv_list[k], reverse=True)
    rate_nodes_dict = {}
    for i in range(len(rate_nodes)):
        rate_nodes_dict[rate_nodes[i]] = i

    adj_full_indices = []
    adj_full_indptr = []
    adj_full_indptr.append(0)
    index_full = 0
    for v in rate_nodes:
        v_s_full = adj_full.indptr[v]
        v_e_full = adj_full.indptr[v + 1]
        for col in adj_full.indices[v_s_full:v_e_full]:
            adj_full_indices.append(rate_nodes_dict[col])
        data_col = adj_full_indices[v_s_full:v_e_full]
        index_full += v_e_full - v_s_full
        adj_full_indptr.append(index_full)

    adj_full.indices = np.array(adj_full_indices, dtype=np.int32)
    adj_full.indptr = np.array(adj_full_indptr, dtype=np.int32)
    
    feats_reorder = feats[rate_nodes]
    class_data_reorder = torch.tensor(class_data[rate_nodes])

    train_idx_reorder = []
    valid_idx_reorder = []
    test_idx_reorder = []
    for i in range(len(rate_nodes)):
        if i < train_idx.shape[0]:
            train_idx_reorder.append(rate_nodes_dict[i])
        elif i < train_idx.shape[0] + valid_idx.shape[0]:
            valid_idx_reorder.append(rate_nodes_dict[i])
        else:
            test_idx_reorder.append(rate_nodes_dict[i])
    train_idx_reorder = torch.tensor(train_idx_reorder)
    valid_idx_reorder = torch.tensor(valid_idx_reorder)
    test_idx_reorder = torch.tensor(test_idx_reorder)

    return adj_full, feats_reorder, class_data_reorder, train_idx_reorder, valid_idx_reorder, test_idx_reorder

def get_order_neighbors(lap_matrix, nodes, num_conv_layers):
    cur_nodes = nodes
    for i in range(num_conv_layers):
        cur_nodes = np.unique(np.concatenate((get_neighbors(lap_matrix, cur_nodes), cur_nodes)))
        #cur_nodes = list(neighbors)
    return cur_nodes

def pagraph(train_nodes, lap_matrix, sample_prob, devices, feat_data, num_devs, num_conv_layers, num_nodes_per_dev, nblocks=20):
    device_id_of_nodes_group = []
    idx_of_nodes_on_device_group = []
    gpu_buffer_group = [-1] * num_devs
    nodes_set_list = [] # In Algorithm1: The temple nodes set on each gpu
    PV = [1] * num_devs # In Algorithm1: The number of nodes including repeated nodes on gpu
    score =[0] * num_devs # In Algorithm1: The score of each gpu
    train_nodes_set = []

    block_size = len(train_nodes) // nblocks

    # Initialize 
    #nodes_set = set()
    device_id_of_nodes = np.array([-1] * lap_matrix.shape[1])
    for i in range(num_devs):
        device_id_of_nodes_group.append(device_id_of_nodes.copy())
    idx_of_nodes_on_device = np.arange(lap_matrix.shape[1])
    idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs

    # Algorithm1: Allocate nodes and their neighbors to different gpu
    for i in range(num_devs):
        batch_nodes = train_nodes[i*block_size: (i+1)*block_size]
        nodes_set = get_order_neighbors(lap_matrix, batch_nodes, num_conv_layers)
        PV[i] += len(nodes_set)
        nodes_set_list.append(nodes_set)
        train_nodes_set.append(batch_nodes)
    for j in range(num_devs * block_size, len(train_nodes), block_size):
        batch_nodes = train_nodes[j: min(j+block_size, len(train_nodes))]
        nodes_set = get_order_neighbors(lap_matrix, batch_nodes, num_conv_layers)
        for i in range(num_devs):
            score[i] = len(np.intersect1d(nodes_set_list[i], nodes_set, assume_unique=True)) * (lap_matrix.shape[0] - len(nodes_set_list[i])) / PV[i]
        max_score_device = score.index(max(score, key=abs))
        PV[max_score_device] += len(nodes_set)
        nodes_set_list[max_score_device] = np.unique(np.concatenate((nodes_set_list[max_score_device], nodes_set)))
        train_nodes_set[max_score_device] = np.concatenate((train_nodes_set[max_score_device], batch_nodes))

    # Save the top buffer_size nodes on each gpu
    for i in range(num_devs):
        gpu_buffer_group[i] = list(map(list(sample_prob[list(nodes_set_list[i])]).index, heapq.nlargest(num_nodes_per_dev, sample_prob[list(nodes_set_list[i])])))
        device_id_of_nodes_group[i][gpu_buffer_group[i][:]] = devices[i]
        idx_of_nodes_on_device_group[i][gpu_buffer_group[i][:]] = range(num_nodes_per_dev)
    
    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set

def create_buffer(lap_matrix, graph_data, num_nodes_per_dev, devices, dataset, num_conv_layers, alpha=1, pagraph_partition=False, naive_partition=False, random_partition=False):
    
    _, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data
    
    num_devs = len(devices)

    fname = f'save/{dataset}.({num_devs}).({num_nodes_per_dev}).({alpha}).({num_conv_layers}).({pagraph_partition}).({naive_partition}).({random_partition})buf'

    train_nodes_set = None


    if not os.path.exists(fname):
        # node original id on each device
        gpu_buffer_group = []
        # node idx on each device
        idx_of_nodes_on_device_group = []
        device_id_of_nodes_group = []
        if naive_partition == True:
            idx_of_nodes_on_device = np.arange(lap_matrix.shape[1])
            if random_partition == True:
                shuffle(idx_of_nodes_on_device)
            device_id_of_nodes = np.array([-1] * lap_matrix.shape[1])
            for i in range(num_devs):
                buffered_nodes_on_dev_i = idx_of_nodes_on_device[i*num_nodes_per_dev : (i+1)*num_nodes_per_dev]
                print(len(buffered_nodes_on_dev_i))
                gpu_buffer_group.append(buffered_nodes_on_dev_i)
                device_id_of_nodes[buffered_nodes_on_dev_i] = devices[i]
                idx_of_nodes_on_device[buffered_nodes_on_dev_i] = np.arange(len(buffered_nodes_on_dev_i))
            device_id_of_nodes_group = [device_id_of_nodes] * num_devs
            idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs
            pickle.dump([device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group], open(fname, 'wb'))
        else:
            sample_prob = np.ones(len(train_nodes)) * lap_matrix[train_nodes, :]
            for i in range(num_conv_layers-1):
                sample_prob *= lap_matrix
            buffer_size = num_nodes_per_dev * num_devs
            buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]

            if pagraph_partition == True:
                device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set = pagraph(train_nodes, lap_matrix, sample_prob, devices, feat_data, num_devs, num_conv_layers, num_nodes_per_dev)

                pickle.dump([device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set], open(fname, 'wb'))
            else:
                idx_of_nodes_on_device = np.arange(lap_matrix.shape[1])
                for i in range(num_devs):
                    device_id_of_nodes = np.array([-1] * lap_matrix.shape[1])
                    gpu_buffer_group.append(buffered_nodes[:num_nodes_per_dev].copy())
                    buffered_nodes_on_dev_i = buffered_nodes[:num_nodes_per_dev]
                    device_id_of_nodes[buffered_nodes_on_dev_i] = devices[i]
                    device_id_of_nodes_group.append(device_id_of_nodes.copy())
                    idx_of_nodes_on_device[buffered_nodes_on_dev_i] = np.arange(len(buffered_nodes_on_dev_i))    
                
                idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs

                p_accum = np.array([0.0] * num_devs)
                for i in range(len(buffered_nodes) - num_nodes_per_dev):
                    if i % (num_devs-1) == 0:
                        device_order = np.argsort(p_accum)

                    candidate_node = buffered_nodes[num_nodes_per_dev + i]
                    new_node_idx = num_nodes_per_dev - 1 - i // (num_devs-1)

                    node_to_be_replaced = buffered_nodes[new_node_idx]
                    if sample_prob[candidate_node] >= alpha * sample_prob[node_to_be_replaced]:
                        current_dev = device_order[i % (num_devs-1)]
                        p_accum[current_dev] += sample_prob[candidate_node]
                        for j in range(num_devs):
                            device_id_of_nodes_group[j][candidate_node] = devices[current_dev]
                            idx_of_nodes_on_device_group[j][candidate_node] = new_node_idx
                        device_id_of_nodes_group[current_dev][node_to_be_replaced] = devices[device_order[-1]] 
                        gpu_buffer_group[current_dev][new_node_idx] = candidate_node 
                    else:
                        break
                change_num = i

                pickle.dump([change_num, p_accum, device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group], open(fname, 'wb'))
    else:
        if pagraph_partition == True:
            device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set = pickle.load(open(fname, 'rb'))
        elif naive_partition == True:
            device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group = pickle.load(open(fname, 'rb'))
        else:
            change_num, p_accum, device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group = pickle.load(open(fname, 'rb'))
            print(p_accum)
            print("change_num: ", change_num)

    gpu_buffers = []
    for i in range(num_devs):
        gpu_buffers.append(feat_data[gpu_buffer_group[i]].to(devices[i]))

    # gpu_buffer_group: the actual training node index on each GPU. shape:[number of GPUs][number of nodes on each GPU]
    # gpu_buffers: the feature of the nodes on each GPU
    # device_id_of_nodes_group: the device of each nodes on each GPU
    # 
    #print(device_id_of_nodes_group[0][1:1000])
    #print(device_id_of_nodes_group[1][1:1000])
    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers, gpu_buffer_group, train_nodes_set


def get_neighbors(adj_matrix, nodes):
    return np.where((np.ones(len(nodes)) * adj_matrix[nodes, :]) != 0)[0]


def get_skewed_sampled_nodes(adj_matrix, gpu_buffers_group, orders):
    neighboring_nodes = []
    neighboring_nodes.append(np.unique(np.concatenate(gpu_buffers_group)))
    v = np.array([0]*adj_matrix.shape[1])
    v[neighboring_nodes[0]] = 1

    for i in range(1, len(orders)):
        v = v * adj_matrix
        neighboring_nodes.append(np.argsort(-1*v)[:8192])
    return neighboring_nodes
    


def create_shared_input_object(lap_matrix, graph_data):
    lap_matrix_indptr = mp.Array('l', lap_matrix.indptr)
    lap_matrix.indptr = None
    lap_matrix_indices = mp.Array('l', lap_matrix.indices)
    lap_matrix.indices = None
    lap_matrix_data = mp.Array('f', lap_matrix.data)
    lap_matrix.data = None

    class_arr_indptr = mp.Array('l', graph_data[1].indptr)
    graph_data[1].indptr = None
    class_arr_indices = mp.Array('l', graph_data[1].indices)
    graph_data[1].indices = None
    class_arr_data = mp.Array('i', graph_data[1].data)
    graph_data[1].data = None



    res = [(lap_matrix_indptr, lap_matrix_indices, lap_matrix_data, lap_matrix.shape), (class_arr_indptr, class_arr_indices, class_arr_data, graph_data[1].shape), *graph_data[2:]]
    print("shared object created", flush=True)
    return res

    

    

