from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import torch.distributed as dist
import subprocess
from itertools import groupby 
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected



def load_graphsaint_data(graph_name, root_dir):
    # adj_full: graph edges stored in coo format, role: dict storing indices of train, val, test nodes
    # feats: features of all nodes, class_map: label of all nodes
    adj_full = sp.load_npz(f'{root_dir}/{graph_name}/adj_full.npz').astype(np.float)
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
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = sp.lil_matrix((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = sp.lil_matrix((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k, v-offset] = 1
    
    print('feat dim: ', feats.shape, flush=True)
    print('label dim: ', class_arr.shape, flush=True)
    

    return (adj_full, class_arr, torch.FloatTensor(feats).pin_memory(), num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))

def load_ogbn_data(graph_name, root_dir):
    dataset = PygNodePropPredDataset(graph_name, root=root_dir)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    num_vertices = data.num_nodes
    adj_full = sp.csr_matrix(([1]*len(row), (row, col)), shape=(num_vertices, num_vertices))
    feats = data.x.pin_memory()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    class_data = data.y.data.flatten()
    assert(len(class_data) == num_vertices)

    print(class_data)

    num_classes = max(class_data) - min(class_data) + 1
    print(num_classes)
    num_classes = int(num_classes.item())
    print(num_classes)
    class_arr = sp.lil_matrix((num_vertices, num_classes))
    offset = min(class_data)
    for i in range(len(class_data)):
        class_arr[i, class_data[i]-offset] = 1

    print('feat dim: ', feats.shape, flush=True)
    print('label dim: ', class_arr.shape, flush=True)
    
    return (adj_full, class_arr, feats, num_classes, train_idx.numpy(), valid_idx.numpy(), test_idx.numpy())

    

# the columns of sample_matrix must be all nodes
def create_buffer(train_data, num_nodes_per_dev, devices, alpha=1):
    
    lap_matrix, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = train_data

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
    # for example num_devs=4, num_nodes_per_dev=m
    for i in range(len(buffered_nodes) - num_nodes_per_dev):
        candidate_node = buffered_nodes[num_nodes_per_dev + i]
        node_to_be_replaced = buffered_nodes[num_nodes_per_dev - 1 - i // (num_devs - 1)]
        if sample_prob[candidate_node] > alpha * sample_prob[node_to_be_replaced]:
            for j in range(num_devs):
                device_id_of_nodes_group[j][candidate_node] = devices[i % num_devs]
                idx_of_nodes_on_device_group[j][candidate_node] = num_nodes_per_dev - 1 - i // (num_devs - 1)
            device_id_of_nodes_group[i % num_devs][node_to_be_replaced] = num_devs - 1 - i //  (num_devs - 1) % num_devs
            gpu_buffer_group[i % num_devs][num_nodes_per_dev - 1 - i // (num_devs - 1)] = candidate_node 
        else:
            break
            
    print("change_num: ", i)

    gpu_buffers = []
    for i in range(num_devs):
        gpu_buffers.append(feat_data[gpu_buffer_group[i]].to(devices[i]))

    print(gpu_buffer_group)

    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers

