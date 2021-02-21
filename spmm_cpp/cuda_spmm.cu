#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

static const int NNZ_PER_CHUNK = 32;
static const int SROW_PER_TILE = 2;
static const int LENGTH = (256 / SROW_PER_TILE);

#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CUDA_CALL(X) ERR_NE((X), cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

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

torch::Tensor _to_csr_int(const torch::Tensor &rowIndices, int64_t dim) {
  auto options = rowIndices.options().dtype(torch::kInt);
  torch::Tensor csr = torch::empty({dim + 1}, options);
  Xcoo2csr(rowIndices.data_ptr<int32_t>(), rowIndices.size(0), dim, csr.data_ptr<int32_t>());
  return csr;
}

inline int DIV(int x, int tile_size) {
	if (x % tile_size) return x / tile_size + 1; else return x / tile_size; } 

}  // namespace

__global__ void _spmm_cuda_kernel(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> colidx,

    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int64_t nrows,
    const int64_t ncols,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> denseMat,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> resMat) {
	
	__shared__ float values_buf[SROW_PER_TILE][LENGTH];
	__shared__ int colidx_buf[SROW_PER_TILE][LENGTH];
  int r = blockIdx.x * SROW_PER_TILE + threadIdx.y;
  if (r < rowptr.size(0) - 1) {
		int c = blockIdx.y * 64 + threadIdx.x;
		int c2 = c + 32;

		float pares = 0.0;
		float pares2 = 0.0;

		for (int b=rowptr[r]; b<rowptr[r+1]; b+=LENGTH) {
			int length = LENGTH > rowptr[r+1] - b ? rowptr[r+1] - b : LENGTH;
			for (int i=threadIdx.x; i<length; i+=32) {
				values_buf[threadIdx.y][i] = values[i+b];
				colidx_buf[threadIdx.y][i] = colidx[i+b];
			}
			for (int k=0; k<length; k++) {
				pares += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c];
				pares2 += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c];
			}
		}
		if (rowptr[r+1] - rowptr[r] > 0) {
			resMat[r][c] = pares;
			resMat[r][c2] = pares2;
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

  //	std::cout << nbx << " " << nby << std::endl;

  dim3 nthreads(32, SROW_PER_TILE);
  dim3 nblocks(DIV(rowptr.size(0), SROW_PER_TILE), DIV(denseMat.size(1), 64));

  //std::cout << resMat.sizes() << std::endl;

  _spmm_cuda_kernel<<<nblocks, nthreads>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

  //cudaDeviceSynchronize();

  return resMat;
}
