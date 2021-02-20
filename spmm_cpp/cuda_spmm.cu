#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <vector>
#include <iostream>



static const int NNZ_PER_CHUNK = 32;

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)


namespace {
	void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
		TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
				"cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
				INT_MAX);

		int i_nnz = (int)nnz;
		int i_m = (int)m;
		cusparseHandle_t handle;
		CUSPARSE_CALL(cusparseCreate(&handle));
		CUSPARSE_CALL(cusparseXcoo2csr(handle, coorowind, i_nnz, i_m, csrrowptr, CUSPARSE_INDEX_BASE_ZERO));

		cusparseDestroy(handle);
	}



	torch::Tensor _to_csr_int(const torch::Tensor& rowIndices, int64_t dim) {
		auto options = rowIndices.options().dtype(torch::kInt);
		torch::Tensor csr = torch::empty({dim+1}, options);
		Xcoo2csr(rowIndices.data_ptr<int32_t>(), rowIndices.size(0), dim, csr.data_ptr<int32_t>());
		return csr;
	}
}

__global__ void _spmm_cuda_kernel(
		const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr, 
		const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx, 
		const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> colidx, 

		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values, 
		const int64_t nrows,
		const int64_t ncols,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> denseMat,
		torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> resMat
		) {


	for (int didx = threadIdx.y + blockIdx.y * blockDim.y; didx < ncols; didx += blockDim.y * gridDim.y) {
		for (int vridx = threadIdx.x + blockIdx.x * blockDim.x; vridx < rowptr.size(0) - 1; vridx += blockDim.x * gridDim.x) {
			int ridx = vridx;
			float res = 0.0;
			for (int i=rowptr[vridx]; i<rowptr[vridx+1]; i++) {
				int c = colidx[i];
				res += values[i] * denseMat[c][didx];
			}
			atomicAdd(&resMat[ridx][didx], res);
		}
	}
}

torch::Tensor spmm_cuda(torch::Tensor sparseMat, torch::Tensor denseMat) {

	auto rowidx = sparseMat.indices()[0].to(at::ScalarType::Int);
	auto colidx = sparseMat.indices()[1].to(at::ScalarType::Int);
	auto values = sparseMat.values();


	at::DeviceGuard g(denseMat.device());

	torch::Tensor resMat = torch::zeros({sparseMat.size(0), denseMat.size(1)}, denseMat.options());
	torch::Tensor rowptr = _to_csr_int(rowidx, sparseMat.size(0));	

	cudaDeviceSynchronize();


	//std::cout << rowptr << std::endl;

	//std::cout << resMat.device() << std::endl;

	// get the virtual rowptr
	/*
	   int nrows_a_virtual = 0;
	   int rowptr_size = nrow_a + nnz_a / NNZ_PER_CHUNK + 1;
	   torch::Tensor rowptr = torch::empty({rowptr_size}, indices.options());

	   rowptr[0] = 0;
	   int64_t pre_r = -1;
	   int cur = 0;
	   for (int i = 0; i< indices.size(1); i++) {
	   int64_t ridx = indices[0][i].item<int64_t>();
	   if (ridx == pre_r) {
	   cur += 1;
	   if (cur >= NNZ_PER_CHUNK) {
	   rowptr[nrows_a_virtual] = i;
	   cur = 0;
	   nrows_a_virtual++;
	   }
	   } else {
	   rowptr[nrows_a_virtual] = i; 	
	   pre_r = ridx;
	   cur = 0;
	   nrows_a_virtual++;
	   }
	   }
	   rowptr[nrows_a_virtual] = indices.size(1);*/

	/*for (int i=0; i<nrows_a_virtual+1; i++) {
	  std::cout << rowptr[i] << std::endl;
	  }*/


	int nbx = (rowptr.size(0) - 1) / 16;
	if ((rowptr.size(0) - 1) % 16) nbx++; 
	if (nbx > 32) nbx = 32;
	int nby = denseMat.size(1) / 32;
	if (denseMat.size(1) % 32) nby++;
	if (nby > 32) nby = 32;

	//	std::cout << nbx << " " << nby << std::endl;

	dim3 nthreads(16, 32);
	dim3 nblocks(nbx, nby);

	//std::cout << resMat.sizes() << std::endl;

	_spmm_cuda_kernel<<<nthreads, nblocks>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	//cudaDeviceSynchronize();

	return resMat;
}
