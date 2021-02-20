#include <torch/extension.h>

torch::Tensor spmm_cuda(torch::Tensor sparseMat, torch::Tensor denseMat);

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

torch::Tensor spmm(torch::Tensor sparseMat, torch::Tensor denseMat) {
	CHECK_SPARSE(sparseMat);
	CHECK_DENSE(denseMat);
	return spmm_cuda(sparseMat, denseMat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("spmm", &spmm, "Sparse-Dense Matrix Multiplication");
}
