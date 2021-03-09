from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import torch.distributed as dist
import subprocess
from itertools import groupby 



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

    return (adj_full, class_arr, torch.FloatTensor(feats).pin_memory(), num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))


def get_sample_matrix(adj_matrix, train_nodes, orders, rank, world_size):
    print("creating sample matrix", flush=True)
    adj_matrix_new = adj_matrix + sp.eye(adj_matrix.shape[0])
    num_layers = sum(orders)
    chunk_size = len(train_nodes) // world_size
    if (len(train_nodes) % world_size):
        chunk_size += 1
    chunk_start = rank * chunk_size
    chunk_end = min((rank+1)*chunk_size, len(train_nodes))
    train_nodes_chunk = train_nodes[chunk_start:chunk_end]
    sample_matrix = adj_matrix_new[train_nodes_chunk,:]
    for i in range(num_layers-1):
        sample_matrix = sample_matrix * adj_matrix_new

    assert(len(train_nodes_chunk) == sample_matrix.shape[0])

    avg = np.sum(sample_matrix) // sample_matrix.nnz
    sample_matrix = sample_matrix.multiply(sample_matrix > avg)
    sample_matrix = sample_matrix.tocoo()
    #print(sample_matrix.shape)
    #print(sample_matrix.col)
    #print(sample_matrix.row)
    return {(train_nodes_chunk[sample_matrix.row[i]], sample_matrix.col[i]): 0 for i in range(len(sample_matrix.row))}





# the columns of sample_matrix must be all nodes
def create_buffer(train_data, buffer_size, devices, method='partition'):
    
    adj_matrix, class_arr, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = train_data

    sample_prob = np.ones(len(train_nodes)) * adj_matrix[train_nodes, :] * adj_matrix

    buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]

    num_devs = len(devices)
    num_nodes_per_dev = len(buffered_nodes) // num_devs
    if len(buffered_nodes) % num_devs != 0:
        num_nodes_per_dev += 1

    if method == 'partition':
        gpu_buffers = []
        device_id_of_nodes = np.array([-1] * adj_matrix.shape[1])
        idx_of_nodes_on_device = np.arange(adj_matrix.shape[1])
        for i in range(num_devs):
            start = i * num_nodes_per_dev
            end = min(start + num_nodes_per_dev, len(buffered_nodes))
            buffered_nodes_on_dev_i = buffered_nodes[start:end]
            gpu_buffers.append(feat_data[buffered_nodes_on_dev_i].to(devices[i]))
            device_id_of_nodes[buffered_nodes_on_dev_i] = devices[i]
            idx_of_nodes_on_device[buffered_nodes_on_dev_i] = np.arange(len(buffered_nodes_on_dev_i))

        
        device_id_of_nodes_group = [device_id_of_nodes] * num_devs
        idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs
    
    elif method == 'identical':
        gpu_buffers = []
        device_id_of_nodes_group = []
        device_id_of_nodes = np.array([-1] * adj_matrix.shape[1])
        idx_of_nodes_on_device = np.arange(adj_matrix.shape[1])
        for i in range(num_devs):
            buffered_nodes_on_dev_i = buffered_nodes[:num_nodes_per_dev]
            gpu_buffers.append(feat_data[buffered_nodes_on_dev_i].to(devices[i]))
            device_id_of_nodes[buffered_nodes_on_dev_i] = devices[i]
            device_id_of_nodes_group.append(device_id_of_nodes)
            idx_of_nodes_on_device[buffered_nodes_on_dev_i] = np.arange(len(buffered_nodes_on_dev_i))
        
        idx_of_nodes_on_device_group = [idx_of_nodes_on_device] * num_devs

    return device_id_of_nodes_group, idx_of_nodes_on_device_group, gpu_buffers


def update_sampled_matrix(output_nodes, sampled_cols, sp_mat):
    for r in output_nodes:
        for sc in sampled_cols:
            for c in sc:
                if (r,c) in sp_mat:
                    sp_mat[(r,c)] += 1
    return sp_mat


def reorder_and_restart(adj_matrix, train_nodes, sp_mat, rank, world_size):
    if rank == 0:
        with open('reorder/.tmp_sp_mat.txt', 'w') as f: 
            f.write(f'{len(train_nodes)} {adj_matrix.shape[1]}''\n')
    dist.barrier()
    with open('reorder/.tmp_sp_mat.txt', 'a') as f:
        for k, v in sp_mat.items():
            if v > 1:
                f.write(f'{k[0]} {k[1]} {v}''\n')
    dist.barrier()
    if rank == 0:
        subprocess.run(["make", "-C", "reorder"])
        subprocess.run(["reorder/reorder", "reorder/.tmp_sp_mat.txt", "reorder/.tmp_reorder.txt"])
    dist.barrier()
    #print(rank, token.item())
    with open('reorder/.tmp_reorder.txt', 'r') as f:
        len1 = len(train_nodes) 
        train_nodes = np.array([int(i) for i in f.readlines()])
        assert(len(np.unique(train_nodes))==len1)
        #print(train_nodes)
    chunk_size = len(train_nodes) // world_size
    if (len(train_nodes) % world_size):
        chunk_size += 1
    chunk_start = rank * chunk_size
    chunk_end = min((rank+1)*chunk_size, len(train_nodes))
    train_nodes_chunk = set(train_nodes[chunk_start:chunk_end])
    sp_prob = np.zeros(adj_matrix.shape[1])
    for k, v in sp_mat.items():
        if k[0] in train_nodes_chunk:
            sp_prob[k[1]] += v
    
    return sp_prob

# TODO
def collect_samples(sp_mat_coo, output_nodes, sampled_cols):
    for sc in sampled_cols:
        x, y = np.meshgrid(output_nodes, sc)
        x = x.flatten()
        y = y.flatten()
        sp_mat_coo.extend(list(zip(x, y)))

    #(k, sum(1 for x in v)) for k, v in groupby(sorted(zip(x,y)))]

# TODO
def merge_samples(sp_mat_coo, nrows, ncols, train_nodes, rank, world_size):
    if rank == 0:
        with open('reorder/.sp_mat_coo.txt', 'w') as f:
            f.write(f'{nrows} {ncols}''\n')
    dist.barrier()

    with open('reorder/.sp_mat_coo.txt', 'a') as f:
        for k, v in groupby(sorted(sp_mat_coo)):
            c = sum(1 for x in v)
            f.write(f'{k[0]} {k[1]} {c}''\n')
    dist.barrier()

    if world_size > 1 and rank == 0:
        subprocess.run(["make", "-C", "reorder"])
        subprocess.run(["reorder/reorder", "reorder/.sp_mat_coo.txt", "reorder/.reordered_nodes.txt"])
    dist.barrier()

    if world_size > 1:
        with open('reorder/.reordered_nodes.txt', 'r') as f:
            train_nodes = np.array([int(i) for i in f.readlines()])
            assert(len(np.unique(train_nodes)) == nrows)
        chunk_size = len(train_nodes) // world_size
        if (len(train_nodes) % world_size):
            chunk_size += 1
        chunk_start = rank * chunk_size
        chunk_end = min((rank+1)*chunk_size, len(train_nodes))
        train_nodes_chunk = set(train_nodes[chunk_start:chunk_end])
        sp_prob = np.zeros(ncols)
        with open('reorder/.sp_mat_coo.txt', 'r') as f:
            for l in f.readlines():
                a, b, c = map(int, l.split())
                if a in train_nodes_chunk:
                    sp_prob[b] += c

    else:
        sp_prob = np.zeros(ncols)
        with open('reorder/.sp_mat_coo.txt', 'r') as f:
            for l in f.readlines():
                a, b, c = map(int, l.split())
                sp_prob[b] += c
        

    return train_nodes, sp_prob