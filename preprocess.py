from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *

def prepare_data(pool, sampler, target_nodes, samp_num_list, num_nodes, lap_matrix, orders, batch_size, mode='train'):
    if mode == 'train' or mode == 'test':
        # sample p batches for training
        idxs = torch.randperm(len(target_nodes))
        num_batches = len(target_nodes) // batch_size
        if (len(target_nodes) % batch_size):
            num_batches += 1
        for i in range(0, num_batches, 32):   # 32 is the queue size
            futures = []
            for j in range(i, min(32+i, num_batches)):
                futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), target_nodes[idxs[j*batch_size: min((j+1)*batch_size, len(idxs))]], samp_num_list, num_nodes, lap_matrix, orders))
            yield from futures
    elif mode == 'val':
        futures = []
        # sample a batch with more neighbors for validation
        idx = torch.randperm(len(target_nodes))[:batch_size]
        batch_nodes = target_nodes[idx]
        futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, orders))
        yield from futures


# the columns of sample_matrix must be all nodes
def create_buffer(sample_prob, feat_data, buffer_size, device):
    #print(col_sum)
    #print(args.buffer_size)
    buffered_nodes = np.argsort(-1*sample_prob)[:buffer_size]
    buffer = feat_data[buffered_nodes].to(device)
    print('GPU buffer created, size: ', len(buffered_nodes))
    buffer_map = np.arange(len(sample_prob))
    buffer_map[buffered_nodes] = np.arange(len(buffered_nodes))
    buffer_mask = np.array([False] * len(sample_prob))
    buffer_mask[buffered_nodes] = True
    return buffer, buffer_map, buffer_mask