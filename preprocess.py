from concurrent.futures import ThreadPoolExecutor, as_completed
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
import metis


def partition_graph(graph_data, devices):

    _, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data

    N = feat_data.shape[0]
    n = N // len(devices)
    if N % len(devices): 
        n += 1

    device_id_of_nodes_group = []

    device_id_of_nodes = np.array([-1] * N)
    idx_of_nodes_on_device = np.array([-1] * N)

    gpu_buffer_group = []



    for i in range(len(devices)):
        start = i * n
        end = N if (i+1) * n > N else (i+1) * n 
        device_id_of_nodes[start: end] = devices[i]
        idx_of_nodes_on_device[start: end] = np.arange(end-start)
        gpu_buffer_group.append(np.arange(start, end))
    

    for i in range(len(devices)):
        device_id_of_nodes_group.append(device_id_of_nodes.copy())


    idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * len(devices)
    print(gpu_buffer_group)


    gpu_buffers = []
    for i in range(len(devices)):
        gpu_buffers.append(feat_data[gpu_buffer_group[i]].to(devices[i]))

    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers


def load_graphsaint_data(graph_name, root_dir, num_devs):
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

    # reorder graph
    adj_full, feats, class_map, role = reorder_graphsaint_graph(adj_full, feats, class_map, role, num_devs)

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


def load_ogbn_data(graph_name, root_dir, num_devs):
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

    #reorder graph
    adj_full, feats, class_data, train_idx, valid_idx, test_idx = reorder_ogbn_graph(adj_full, feats, class_data, train_idx, valid_idx, test_idx, num_devs)


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


    np.savetxt('tmp.output', sp.linalg.norm(adj_full, ord=0, axis=0))
    
    return (adj_full, class_arr, feats, num_classes, train_idx, valid_idx, test_idx)


def reorder_graphsaint_graph(adj_full, feats, class_map, role, num_devs):
    pv_list = [
            adj_full.indices[
                adj_full.indptr[v] : adj_full.indptr[v + 1]
            ] 
            for v in range(adj_full.shape[0])
        ]
    
    (edgecuts, parts) = metis.part_graph(pv_list, num_devs)
    #print(parts)
    rate_nodes = sorted(range(len(pv_list)), key=lambda k: parts[k], reverse=True)

    rate_nodes_dict = [0] * adj_full.shape[0] 
    for i in range(len(rate_nodes)):
        rate_nodes_dict[rate_nodes[i]] = i

    adj_full_indices = []
    adj_full_indptr = []
    adj_full_indptr.append(0)
    index_full = 0
    index_train = 0
    for v in rate_nodes:
        v_s_full = adj_full.indptr[v]
        v_e_full = adj_full.indptr[v + 1]
        for col in adj_full.indices[v_s_full:v_e_full]:
            adj_full_indices.append(rate_nodes_dict[col])
        index_full += v_e_full - v_s_full
        adj_full_indptr.append(index_full)

    adj_full.indices = np.array(adj_full_indices, dtype=np.int32)
    adj_full.indptr = np.array(adj_full_indptr, dtype=np.int32)
    
    feats_reorder = feats[rate_nodes]
    class_map_reorder = {}
    role_reorder = {}
    for i in range(adj_full.shape[0]):
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

    return adj_full, feats_reorder, class_map_reorder, role_reorder
    
def reorder_ogbn_graph(adj_full, feats, class_data, train_idx, valid_idx, test_idx, num_devs):

    pv_list = [
            adj_full.indices[
                adj_full.indptr[v] : adj_full.indptr[v + 1]
            ]
            for v in range(adj_full.shape[0])
        ]

    (edgecuts, parts) = metis.part_graph(pv_list, num_devs)

    # current idx ==> original idx
    rate_nodes = sorted(range(len(pv_list)), key=lambda k: parts[k], reverse=True)
    rate_nodes_dict = np.array([0] * adj_full.shape[0])

    # original idx ==> current idx
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

        index_full += v_e_full - v_s_full
        adj_full_indptr.append(index_full)

    adj_full.indices = np.array(adj_full_indices, dtype=np.int32)
    adj_full.indptr = np.array(adj_full_indptr, dtype=np.int32)
    
    feats_reorder = feats[rate_nodes]
    class_data_reorder = torch.tensor(class_data[rate_nodes])

    train_idx_reorder = rate_nodes_dict[train_idx]
    valid_idx_reorder = rate_nodes_dict[valid_idx]
    test_idx_reorder = rate_nodes_dict[test_idx]


    return adj_full, feats_reorder, class_data_reorder, train_idx_reorder, valid_idx_reorder, test_idx_reorder

def get_order_neighbors(lap_matrix, nodes, num_conv_layers):
    cur_nodes = nodes
    for i in range(num_conv_layers):
        cur_nodes = np.unique(np.concatenate((get_neighbors(lap_matrix, cur_nodes), cur_nodes)))
        #cur_nodes = list(neighbors)
    return cur_nodes

def metis_buffering(train_nodes, lap_matrix, sample_prob, devices, feat_data, num_devs, num_conv_layers, num_nodes_per_dev):
    device_id_of_nodes_group = []
    idx_of_nodes_on_device_group = []
    gpu_buffer_group = [-1] * num_devs
    nodes_set_list = [] # In Algorithm1: The temple nodes set on each gpu
    train_nodes_set = []

    block_size = lap_matrix.shape[1] // len(devices)
    if lap_matrix.shape[1] % len(devices) != 0:
        block_size += 1

    device_id_of_nodes = np.array([-1] * lap_matrix.shape[1])
    for i in range(num_devs):
        device_id_of_nodes_group.append(device_id_of_nodes.copy())
    idx_of_nodes_on_device = np.arange(lap_matrix.shape[1])
    idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs

    
    # Save the top buffer_size nodes on each gpu
    for i in range(num_devs):
        gpu_buffer_group[i] = np.argsort(-1*sample_prob[i*block_size: min((i+1)*block_size, lap_matrix.shape[0])])[:num_nodes_per_dev] + i*block_size
        for j in range(num_devs):
            device_id_of_nodes_group[j][gpu_buffer_group[i][:]] = devices[i]
            idx_of_nodes_on_device_group[j][gpu_buffer_group[i][:]] = range(len(idx_of_nodes_on_device_group[j][gpu_buffer_group[i][:]]))
    
    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set

def create_buffer(lap_matrix, graph_data, num_nodes_per_dev, devices, dataset, num_conv_layers):
    
    _, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data
    
    num_devs = len(devices)

    fname = f'save/{dataset}.({num_devs}).({num_nodes_per_dev}).({num_conv_layers}).(metis).buf'

    train_nodes_set = None


    if not os.path.exists(fname):
        sample_prob = np.ones(len(train_nodes)) * lap_matrix[train_nodes, :]

        device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set = metis_buffering(train_nodes, lap_matrix, sample_prob, devices, feat_data, num_devs, num_conv_layers, num_nodes_per_dev)

        pickle.dump([device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set], open(fname, 'wb'))
    
    else:
        device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffer_group, train_nodes_set = pickle.load(open(fname, 'rb'))

    gpu_buffers = []
    for i in range(num_devs):
        gpu_buffers.append(feat_data[gpu_buffer_group[i]].to(devices[i]))

    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers, gpu_buffer_group, train_nodes_set


def get_neighbors(adj_matrix, nodes):
    return np.where((np.ones(len(nodes)) * adj_matrix[nodes, :]) != 0)[0]


def get_skewed_sampled_nodes(adj_matrix, gpu_buffers_group, orders):
    sampled_nodes_group = [[None for j in range(len(orders))] for i in range(len(gpu_buffers_group))]

    for i in range(len(gpu_buffers_group)):
        cur_nodes = gpu_buffers_group[i]
        sampled_nodes_group[i][0] = cur_nodes
        for j in range(1, len(orders)):
            cur_nodes = get_neighbors(adj_matrix, cur_nodes)
            #cur_nodes = np.unique(np.concatenate((get_neighbors(adj_matrix, cur_nodes), cur_nodes)))
            sampled_nodes_group[i][j] = cur_nodes

        #print(sampled_nodes_group[i])
    
    return sampled_nodes_group
    


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

    

    

