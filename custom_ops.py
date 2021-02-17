import torch
import time
import sys

spmm_forward_time = 0.0
spmm_backward_time = 0.0

class SparseDenseMM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mat1, mat2):
    global spmm_forward_time
    t1 = time.time()
    #print('forward: ', mat1.layout, mat1.shape)
    ctx.save_for_backward(mat1, mat2)
    output = mat1.mm(mat2)
    spmm_forward_time += time.time() - t1
    return output

  @staticmethod
  def backward(ctx, grad_output):
    global spmm_backward_time
    t1 = time.time()
    mat1, mat2 = ctx.saved_tensors
    mat1 = mat1.transpose(0, 1)
    #print('backward: ', mat1.layout, mat1.shape)
    grad_mat2 = mat1.mm(grad_output)
    spmm_backward_time += time.time() - t1
    return None, grad_mat2


spmm = SparseDenseMM.apply


