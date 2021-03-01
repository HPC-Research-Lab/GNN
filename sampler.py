from utils import *

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders):
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
    return adjs, previous_nodes, batch_nodes, sampled_nodes

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, orders):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx if i>0 else None for i in orders], np.arange(num_nodes), batch_nodes, [torch.from_numpy(np.arange(num_nodes).astype(np.int64)) for i in orders]