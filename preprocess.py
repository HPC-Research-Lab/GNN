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


def create_adj_buffer(sample_prob, lap_matrix, buffer_size, device):
    buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]
    buffer_map = np.arange(len(sample_prob))
    buffer_map[buffered_nodes] = np.arange(len(buffered_nodes))
    buffer_mask = np.array([False] * len(sample_prob))
    buffer_mask[buffered_nodes] = True 
    buffer = {r: [torch.from_numpy(lap_matrix[r].indices.astype(np.int)).to(device), torch.from_numpy(lap_matrix[r].data.astype(np.float)).to(device)] for r in buffered_nodes}
    return buffer, buffer_map, buffer_mask