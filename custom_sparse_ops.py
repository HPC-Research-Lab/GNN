import torch
import time
import sys
from torch.utils.cpp_extension import load
import numpy as np

spmm_cpp = load(name='spmm', sources=['spmm_cpp/spmm.cpp', 'spmm_cpp/cuda_spmm.cu'])


spmm_forward_time = 0.0
spmm_backward_time = 0.0



class SparseDenseMM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mat1, mat2):
    global spmm_forward_time
    torch.cuda.synchronize()
    t1 = time.time()
    ctx.save_for_backward(mat1)
    output = spmm_cpp.spmm(mat1, mat2)
    #output = mat1.mm(mat2)
    torch.cuda.synchronize()
    spmm_forward_time += time.time() - t1
    return output

  @staticmethod
  def backward(ctx, grad_output):
    global spmm_backward_time
    torch.cuda.synchronize()
    t1 = time.time()
    mat1, = ctx.saved_tensors
    mat1 = mat1.transpose(0, 1)
    grad_mat2 = mat1.mm(grad_output)
    torch.cuda.synchronize()
    spmm_backward_time += time.time() - t1
    return None, grad_mat2



spmm = SparseDenseMM.apply

## Testing
#for i in range(20):
#  print(f"testing {i}")
#  nc = np.random.randint(100, 2000)
#  print(f'nc {nc}')
#  a = torch.randn(np.random.randint(100,2000), nc).to_sparse().cuda().requires_grad_(True)
#  b = torch.randn(nc, np.random.randint(1,2000)).cuda().requires_grad_(True)
  #print(a)
  #print(b)
#  y = spmm(a, b)
  #print(y)
  #print(a.mm(b))
#  print(torch.norm(y - a.mm(b)))
#  assert(torch.norm(y - a.mm(b)) < 0.1)

#print(a.mm(b))

#y.sum().backward()
#print(b.grad)

#from multiprocessing import Pool

#def f(i, j):
#  return i + j


#with Pool(5) as pool:
#  res = pool.starmap(f, [(i, i*i) for i in range(10)])
#  print(res)


from concurrent.futures import ThreadPoolExecutor, as_completed

def f(i, j):
  return i + j


args = [(i, i*i) for i in range(10)]

def gen():
  futures = []
  with ThreadPoolExecutor(max_workers=8) as executor:
    for i in range(10):
      futures.append(executor.submit(f, i, i*i))
  return futures
  

for fut in as_completed(gen()):
  print(fut.result())

