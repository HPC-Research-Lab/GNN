import torch
import time
import sys
import spmm_cpp

spmm_forward_time = 0.0
spmm_backward_time = 0.0



class SparseDenseMM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mat1, mat2):
    global spmm_forward_time
    #print('forward: ', mat1._indices())
    ctx.save_for_backward(mat1)
    t1 = time.time()
    #output = spmm_cpp.spmm(mat1, mat2) 
    output = mat1.mm(mat2) 
    #output = output.reshape(mat2.shape[1], mat1.shape[0]).transpose(0,1)
    #print('forward1: ', output)
    #print('forward2: ', mat1.mm(mat2))
    #print(torch.norm(output - mat1.mm(mat2)))
    spmm_forward_time += time.time() - t1
    return output

  @staticmethod
  def backward(ctx, grad_output):
    global spmm_backward_time
    t1 = time.time()
    mat1, = ctx.saved_tensors
    mat1 = mat1.transpose(0, 1)
    grad_mat2 = mat1.mm(grad_output)
    spmm_backward_time += time.time() - t1
    return None, grad_mat2



spmm = SparseDenseMM.apply


#a = torch.randn(2, 3).to_sparse().cuda().requires_grad_(True)
#b = torch.randn(3, 2).cuda().requires_grad_(True)

#y = spmm(a, b)

#print(y)

#y.sum().backward()
#print(b.grad)