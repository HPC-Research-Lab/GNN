#include <torch/extension.h>

torch::Tensor spmm_cuda_v1(torch::Tensor sparseMat, torch::Tensor denseMat);
torch::Tensor spmm_cuda_v2(torch::Tensor sparseMat, torch::Tensor denseMat);

torch::Tensor spmm_cuda_v3(torch::Tensor row, torch::Tensor col, torch::Tensor value, int nrows, torch::Tensor denseMat);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_COAL(x) \
  TORCH_CHECK(x.is_coalesced(), #x " must be coalesced")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DENSE(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_SPARSE(x) \
  CHECK_CUDA(x);       \
  CHECK_COAL(x)

torch::Tensor spmm_load_balance(torch::Tensor sparseMat, torch::Tensor denseMat) {
	CHECK_SPARSE(sparseMat);
	CHECK_DENSE(denseMat);
	return spmm_cuda_v2(sparseMat, denseMat);
}

torch::Tensor spmm(torch::Tensor row, torch::Tensor col, torch::Tensor value, int nrows, torch::Tensor denseMat) {
	CHECK_DENSE(row);
	CHECK_DENSE(col);
	CHECK_DENSE(value);
	CHECK_DENSE(denseMat);
	return spmm_cuda_v3(row, col, value, nrows, denseMat); 
}


torch::Tensor spmm_naive(torch::Tensor sparseMat, torch::Tensor denseMat) {
	CHECK_SPARSE(sparseMat);
	CHECK_DENSE(denseMat);
	return spmm_cuda_v1(sparseMat, denseMat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("spmm", &spmm, "Sparse-Dense Matrix Multiplication");
	m.def("spmm_load_balance", &spmm_load_balance, "Sparse-Dense Matrix Multiplication");
}
