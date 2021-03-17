from utils import *



def subgraph_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders, device_id_of_nodes, idx_of_nodes_on_device, scale_factor, concat, device, devices):

    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    orders1 = orders[::-1]
    sampled_nodes = []


    #     row-select the lap_matrix (U) by previously sampled nodes
    U = lap_matrix[previous_nodes , :]
    #     Only use the upper layer's neighborhood to calculate the probability.
    pi = sp.linalg.norm(U, ord=0, axis=0)
    if scale_factor > 1:
        nodes_on_this_gpu = (device_id_of_nodes == device)
        pi[nodes_on_this_gpu] = pi[nodes_on_this_gpu] * scale_factor 
    #pi = np.array(np.sum(U.multiply(U), axis=0))[0]
    p = pi / np.sum(pi)
    samp_num_d = samp_num_list[0]
    #while True:
    s_num = np.min([np.sum(p > 0), samp_num_d])
    #     sample the next layer's nodes based on the adaptively probability (p).
    after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
    #     Add output nodes for self-loop
    after_nodes = np.unique(np.concatenate((after_nodes, previous_nodes)))

    #if concat == True: 
    #    p[previous_nodes] = 0  
    adj = U[: , after_nodes].multiply(1/np.clip(s_num * p[after_nodes], 1e-10, 1))
    adj = sparse_mx_to_torch_sparse_tensor(adj.tocoo().astype(np.float32)).to(device).coalesce()

    layer_idx = 0
    for d in range(len(orders1)):
        layer_idx += 1
        if (orders1[d] == 0):
            adjs.append(None)
            sampled_nodes.append([])
        else:
            adjs.append(adj)
            sampled_nodes.append(np.where(np.in1d(after_nodes, previous_nodes))[0])
            break
    
    for d in range(layer_idx, len(orders1)):
        U = lap_matrix[after_nodes , :]
        adj = U[:, after_nodes].multiply(1/np.clip(s_num * p[after_nodes], 1e-10, 1))
        adjs.append(sparse_mx_to_torch_sparse_tensor(adj.tocoo().astype(np.float32)).to(device).coalesce())
        sampled_nodes.append(np.arange(len(after_nodes)))


    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    sampled_nodes.reverse()

    input_nodes_mask_on_devices = []
    nodes_idx_on_devices = []
    input_nodes_devices = device_id_of_nodes[after_nodes]
    input_nodes_mask_on_cpu = (input_nodes_devices == -1) 
    nodes_idx_on_cpu = after_nodes[input_nodes_mask_on_cpu]

    for i in range(len(devices)):
        input_nodes_mask_on_devices.append(input_nodes_devices == devices[i])
        nodes_idx_on_devices.append(idx_of_nodes_on_device[after_nodes[input_nodes_mask_on_devices[i]]].copy())

    return adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, len(after_nodes), batch_nodes, sampled_nodes



def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders, device_id_of_nodes, idx_of_nodes_on_device, scale_factor, concat, device, devices):
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

    nnz = 0
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
        pi = sp.linalg.norm(U, ord=0, axis=0)
        if scale_factor > 1:
            nodes_on_this_gpu = (device_id_of_nodes == device)
            pi[nodes_on_this_gpu] = pi[nodes_on_this_gpu] * scale_factor 
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
        #if concat == True:
        #    p[previous_nodes] = 0  
        adj = U[: , after_nodes].multiply(1/np.clip(s_num * p[after_nodes], 1e-10, 1))
        nnz += adj.nnz
        adj = sparse_mx_to_torch_sparse_tensor(adj.tocoo().astype(np.float32)).to(device).coalesce()
        adjs.append(adj)

        sampled_nodes.append(np.where(np.in1d(after_nodes, previous_nodes))[0])
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    sampled_nodes.reverse()
    #if len(sampling_time) == 1:

    print('nnz: ', nnz)

    input_nodes_mask_on_devices = []
    nodes_idx_on_devices = []
    input_nodes_devices = device_id_of_nodes[previous_nodes]
    input_nodes_mask_on_cpu = (input_nodes_devices == -1) 
    nodes_idx_on_cpu = previous_nodes[input_nodes_mask_on_cpu]

    for i in range(len(devices)):
        input_nodes_mask_on_devices.append(input_nodes_devices == devices[i])
        nodes_idx_on_devices.append(idx_of_nodes_on_device[previous_nodes[input_nodes_mask_on_devices[i]]].copy())

    return adjs, input_nodes_mask_on_devices, input_nodes_mask_on_cpu, nodes_idx_on_devices, nodes_idx_on_cpu, len(previous_nodes), batch_nodes, sampled_nodes


iter_num = 0
def prepare_data(pool, sampler, target_nodes, samp_num_list, num_nodes, lap_matrix, orders, batch_size, rank, world_size, device_id_of_nodes, idx_of_nodes_on_device, device, devices, concat, scale_factor=1, global_permutation=False, mode='train'):
    global iter_num
    if mode == 'train':
        # sample p batches for training
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
            torch.manual_seed(iter_num)
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
                target_nodes_chunk = target_nodes[idxs[chunk_start+j*batch_size: min(chunk_start+(j+1)*batch_size, chunk_end)]]
                futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), target_nodes_chunk, samp_num_list, num_nodes, lap_matrix, orders, device_id_of_nodes, idx_of_nodes_on_device, scale_factor, concat, device, devices))
            yield from futures
    elif mode == 'val':
        futures = []
        # sample a batch with more neighbors for validation
        idx = torch.randperm(len(target_nodes))[:batch_size]
        batch_nodes = target_nodes[idx]
        futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, orders, device_id_of_nodes, idx_of_nodes_on_device, 1, concat, device, devices))
        yield from futures
    elif mode == 'test':
        num_batches = len(target_nodes) // batch_size
        if (num_batches % batch_size):
          num_batches += 1
        for i in range(0, num_batches, 32):   # 32 is the queue size
            futures = []
            for j in range(i, min(32+i, num_batches)):
                target_nodes_chunk = target_nodes[batch_size*j: min((j+1)*batch_size, len(target_nodes))]
                futures.append(pool.submit(sampler, np.random.randint(2**32 - 1), target_nodes_chunk, samp_num_list, num_nodes, lap_matrix, orders, device_id_of_nodes, idx_of_nodes_on_device, scale_factor, concat, device, devices))
            yield from futures

