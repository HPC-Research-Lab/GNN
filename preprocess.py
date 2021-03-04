from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
import torch.distributed as dist
import subprocess


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

    return (adj_full, class_arr, feats, num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))


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
def create_buffer(sample_prob, feat_data, buffer_size, device):
    #print(col_sum)
    #print(args.buffer_size)
    buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]
    buffer = feat_data[buffered_nodes].to(device)
    #print('GPU buffer created, size: ', len(buffered_nodes))
    buffer_map = np.arange(len(sample_prob))
    buffer_map[buffered_nodes] = np.arange(len(buffered_nodes))
    buffer_mask = np.array([False] * len(sample_prob))
    buffer_mask[buffered_nodes] = True
    return buffer, buffer_map, buffer_mask


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