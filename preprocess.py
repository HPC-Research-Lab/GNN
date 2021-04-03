from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import torch.distributed as dist
import subprocess
from itertools import groupby 
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, dropout_adj
import multiprocessing as mp



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
    

    return (adj_full, class_arr, torch.FloatTensor(feats).pin_memory(), num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))



def load_ogbn_data(graph_name, root_dir):
    dataset = PygNodePropPredDataset(graph_name, root=root_dir)
    split_idx = dataset.get_idx_split()
    data = dataset[0]


    row, col = data.edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    num_vertices = data.num_nodes

    adj_full = sp.csr_matrix(([1]*len(row), (row, col)), shape=(num_vertices, num_vertices), dtype=np.float32)
    row = None
    col = None

    feats = data.x.pin_memory()
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

    

# the columns of sample_matrix must be all nodes
def create_buffer(lap_matrix, graph_data, num_nodes_per_dev, devices, alpha=1):
    
    _, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = graph_data

    sample_prob = np.ones(len(train_nodes)) * lap_matrix[train_nodes, :] * lap_matrix
    #print('skewness: ', len(sample_prob) * np.max(sample_prob) / np.sum(sample_prob))

    num_devs = len(devices)

    buffer_size = num_nodes_per_dev * num_devs

    buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]

    gpu_buffer_group = []
    device_id_of_nodes_group = []
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

        if i % num_devs == 0:
            device_order = np.argsort(p_accum)

        candidate_node = buffered_nodes[num_nodes_per_dev + i]
        new_node_idx = num_nodes_per_dev - 1 - i // (num_devs - 1)
        node_to_be_replaced = buffered_nodes[new_node_idx]
        if sample_prob[candidate_node] > alpha * sample_prob[node_to_be_replaced]:
            current_dev = device_order[i % num_devs]
            p_accum[current_dev] += sample_prob[candidate_node]
            for j in range(num_devs):
                device_id_of_nodes_group[j][candidate_node] = devices[current_dev]
                idx_of_nodes_on_device_group[j][candidate_node] = new_node_idx
            device_id_of_nodes_group[current_dev][node_to_be_replaced] = device_order[-1] 
            gpu_buffer_group[current_dev][new_node_idx] = candidate_node 
        else:
            break

    print(p_accum)
            
    print("change_num: ", i)

    gpu_buffers = []
    for i in range(num_devs):
        gpu_buffers.append(feat_data[gpu_buffer_group[i]].to(devices[i]))

    print(gpu_buffer_group)

    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers


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

    

    

