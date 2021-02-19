import torch
import time
import sys
import ctypes
from ctypes import cdll

spmm_forward_time = 0.0
spmm_backward_time = 0.0

_lib = cdll.LoadLibrary('./sparse_dense_mm.so')


def sparse_dense_mm(indices, values, dims, trans, mat2, res):
  brow = mat2.shape[0]
  bcol = mat2.shape[1]
  #print('forward: ', indices.device, indices[0].data_ptr(), indices[1].data_ptr())
  if not trans:
    _lib.do_cusparse_spmm(ctypes.c_longlong(dims[0]), ctypes.c_longlong(dims[1]), ctypes.c_longlong(len(indices[0])), ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(indices[0].data_ptr()), ctypes.c_void_p(indices[1].data_ptr()), ctypes.c_longlong(bcol), ctypes.c_longlong(brow),  ctypes.c_void_p(mat2.data_ptr()), ctypes.c_longlong(dims[0]), ctypes.c_longlong(bcol), ctypes.c_void_p(res.data_ptr()), ctypes.c_bool(False))
  else:
    _lib.do_cusparse_spmm(ctypes.c_longlong(dims[0]), ctypes.c_longlong(dims[1]), ctypes.c_longlong(len(indices[0])), ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(indices[0].data_ptr()), ctypes.c_void_p(indices[1].data_ptr()), ctypes.c_longlong(brow), ctypes.c_longlong(bcol),  ctypes.c_void_p(mat2.data_ptr()), ctypes.c_longlong(dims[1]), ctypes.c_longlong(bcol),  ctypes.c_void_p(res.data_ptr()), ctypes.c_bool(True))


  

class SparseDenseMM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mat1, mat2):
    global spmm_forward_time
    t1 = time.time()
    ctx.save_for_backward(mat1[0], mat1[1], mat1[2], mat2)
    output = torch.FloatTensor(mat1[2][0], mat2.shape[1]).to(mat1[1].device)
    #print(mat1[1].device, mat1[0].device, mat2.device, output.device)
    sparse_dense_mm(mat1[0], mat1[1], mat1[2], False, mat2, output)
    output = output.reshape(mat2.shape[1], mat1[2][0]).transpose(0,1)
    spmm_forward_time += time.time() - t1
    return output

  @staticmethod
  def backward(ctx, grad_output):
    global spmm_backward_time
    t1 = time.time()
    indices, values, dims, mat2 = ctx.saved_tensors
   # print('backward: ', grad_output, indices, values)
    #print(values.device)
    grad_mat2 = torch.FloatTensor(dims[1], grad_output.shape[1]).to(values.device)
    sparse_dense_mm(indices, values, dims, True, grad_output.contiguous(), grad_mat2)
    grad_mat2 = grad_mat2.reshape(grad_output.shape[1], dims[1]).transpose(0,1)
    #grad_mat2 = grad_mat2.transpose(0,1).reshape((dims[1], grad_output.shape[1]))
    spmm_backward_time += time.time() - t1
    return None, grad_mat2



spmm = SparseDenseMM.apply


#b = torch.FloatTensor([[1,2],[3,4], [5,6]]).to('cuda:0').requires_grad_(True)


#y = spmm((torch.cuda.IntTensor([[0,0,0,1,1,1], [0,1,2,0,1,2]]), torch.cuda.FloatTensor([1,2,3,4,5,6]), torch.cuda.LongTensor([2,3])), b)

#print(y)

#print('============')

#y = y.transpose(0,1)
#y.sum().backward()
#print(b.grad)


#print('##############')
#a1 = torch.FloatTensor([[1,2,3],[4,5,6]]).to_sparse().requires_grad_(True)
#b1 = torch.FloatTensor([[1,2],[3,4], [5,6]]).requires_grad_(True)


#y1 = torch.sparse.mm(a1, b1)
#print(y1)

#print('------------')
#y1.sum().backward()
#print(b1.grad)