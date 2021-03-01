from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *


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

    #reordered = np.array(train_nodes)
    #chunk_size = len(reordered) // num_parts
    #if len(reordered) % num_parts:
    #    chunk_size += 1
    
    #return [(adj_full, class_arr, feats, num_classes, reordered[i*chunk_size : min((i+1)*chunk_size, len(reordered))], np.array(role['va']), np.array(role['te'])) for i in range(num_parts)]
    return (adj_full, class_arr, feats, num_classes, np.array(train_nodes), np.array(role['va']), np.array(role['te']))


iter_num = 0
def prepare_data(pool, sampler, target_nodes, samp_num_list, num_nodes, lap_matrix, orders, batch_size, rank, world_size, buffer_map, buffer_mask, scale_factor=1, mode='train'):
    global iter_num
    if mode == 'train' or mode == 'test':
        # sample p batches for training
        torch.manual_seed(iter_num)
        #print(iter_num)
        iter_num += 1
        idxs = torch.randperm(len(target_nodes))
        chunk_size = len(idxs) // world_size
        if (len(idxs) % world_size):
          chunk_size += 1
        chunk_start = rank * chunk_size
        chunk_end = min((rank+1)*chunk_size, len(idxs))
        num_batches = (chunk_end - chunk_start) // batch_size
        #print(num_batches)
        if (num_batches % batch_size):
          num_batches += 1
        for i in range(0, num_batches, 32):   # 32 is the queue size
            futures = []
            for j in range(i, min(32+i, num_batches)):
                futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), target_nodes[idxs[chunk_start+j*batch_size: min(chunk_start+(j+1)*batch_size, chunk_end)]], samp_num_list, num_nodes, lap_matrix, orders, buffer_map, buffer_mask, scale_factor))
            yield from futures
    elif mode == 'val':
        futures = []
        # sample a batch with more neighbors for validation
        idx = torch.randperm(len(target_nodes))[:batch_size]
        batch_nodes = target_nodes[idx]
        futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, orders, buffer_map, buffer_mask, 1))
        yield from futures


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