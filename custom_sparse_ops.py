import torch
import time
import sys
from torch.utils.cpp_extension import load

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



#a = torch.randn(2, 3).to_sparse().cuda().requires_grad_(True)
#b = torch.randn(3, 2).cuda().requires_grad_(True)

#y = spmm(a, b)

#print(y)

#print(a.mm(b))

#y.sum().backward()
#print(b.grad)