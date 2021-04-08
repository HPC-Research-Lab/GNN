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
    ctx.save_for_backward(mat1)
    #torch.cuda.synchronize()
    #t1 = time.time()
    output = spmm_cpp.spmm_load_balance(mat1, mat2)
    #output = spmm_cpp.spmm_naive(mat1, mat2)
    #output = mat1.mm(mat2)
    #torch.cuda.synchronize()
    #spmm_forward_time += time.time() - t1
    return output

  @staticmethod
  def backward(ctx, grad_output):
    global spmm_backward_time
    mat1, = ctx.saved_tensors
    grad_mat2 = spmm_cpp.spmm_load_balance(mat1.transpose(0,1).coalesce(), grad_output.contiguous())
    #grad_mat2 = spmm_cpp.spmm_naive(mat1.transpose(0,1).coalesce(), grad_output.contiguous())
    #grad_mat2 = mat1.transpose(0,1).mm(grad_output)
    return None, grad_mat2


spmm = SparseDenseMM.apply
create_coo_tensor = spmm_cpp.create_coo_tensor


