import torch
import torch.multiprocessing as mp 
from torch.multiprocessing import set_start_method
import numpy as np
import scipy.sparse as sp
from multiprocessing import Manager, Array



if __name__ == "__main__":
    a = np.ones(1024)
    b = Array('f', a)
    a = None
    print(b[0])


