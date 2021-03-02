from utils import *

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders, buffer_map, buffer_mask, scale_factor):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    #t1 = time.time()
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    orders1 = orders[::-1]
    sampled_nodes = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(len(orders1)):
        if (orders1[d] == 0):
            adjs.append(None)
            sampled_nodes.append([])
            continue
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = scipy.sparse.linalg.norm(U, ord=0, axis=0)
        if scale_factor > 1:
            pi[buffer_mask == True] = pi[buffer_mask == True] * scale_factor 
        #pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        samp_num_d = samp_num_list[d]
        #while True:
        s_num = np.min([np.sum(p > 0), samp_num_d])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, previous_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.    
        p[previous_nodes] = 0  
        adj = U[: , after_nodes].multiply(1/np.clip(s_num * p[after_nodes], 1e-10, 1))
            #if adj.nnz < 2e8:
            #    break
            #else:
            #    print('nnz:', adj.nnz)
            #    samp_num_d /= 2
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]

        sampled_nodes.append(np.where(np.in1d(after_nodes, previous_nodes))[0])
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    sampled_nodes.reverse()
    #if len(sampling_time) == 1:
     #   sampling_time[0] += time.time() - t1

    input_nodes_idx = buffer_map[previous_nodes]
    input_nodes_mask = buffer_mask[previous_nodes]
    feat_gpu_idx = input_nodes_mask == True
    feat_cpu_idx = input_nodes_mask == False
    gpu_nodes_idx = input_nodes_idx[feat_gpu_idx]
    cpu_nodes_idx = input_nodes_idx[feat_cpu_idx]
    
    return adjs, gpu_nodes_idx, cpu_nodes_idx, feat_gpu_idx, feat_cpu_idx, batch_nodes, sampled_nodes

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx if i>0 else None for i in orders], np.arange(num_nodes), batch_nodes, [torch.from_numpy(np.arange(num_nodes).astype(np.int64)) for i in orders]


iter_num = 0
def prepare_data(pool, sampler, target_nodes, samp_num_list, num_nodes, lap_matrix, orders, batch_size, rank, world_size, buffer_map, buffer_mask, scale_factor=1, global_permutation=False, mode='train'):
    global iter_num
    if mode == 'train' or mode == 'test':
        # sample p batches for training
        torch.manual_seed(iter_num)
        #print(iter_num)
        iter_num += 1
        chunk_size = len(target_nodes) // world_size
        if (len(target_nodes) % world_size):
          chunk_size += 1
        chunk_start = rank * chunk_size
        chunk_end = min((rank+1)*chunk_size, len(target_nodes))
        num_batches = (chunk_end - chunk_start) // batch_size
        #print(num_batches)
        if global_permutation == True:
            idxs = torch.randperm(len(target_nodes))
        else:
            idxs = torch.LongTensor(len(target_nodes))
            idxs[chunk_start:chunk_end] = torch.randperm(chunk_end-chunk_start) + chunk_start
            #print(idxs)
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
