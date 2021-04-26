#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

static const int NNZ_PER_CHUNK = 64;
static const int SROW_PER_TILE = 2;
static const int LENGTH = (256 / SROW_PER_TILE);

static const int THREADS_PER_BLOCK = 512;
static const int ELEMENTS_PER_BLOCK = 1024;

#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CUDA_CALL(X) ERR_NE((X), cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

#define DIV(x, ts) \
  ((x) % (ts) != 0 ? (x) / (ts) + 1 : (x) / (ts))

#define LOG_MEM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

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

inline int nextPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

}  // namespace

// no load balancing
__global__ void _spmm_cuda_v1_kernel(
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

    if (c2 < ((denseMat.size(1) >> 6) << 6)) {
      float pares = 0.0;
      float pares2 = 0.0;

      for (int b = rowptr[r]; b < rowptr[r + 1]; b += LENGTH) {
        int length = LENGTH > rowptr[r + 1] - b ? rowptr[r + 1] - b : LENGTH;
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
          values_buf[threadIdx.y][i] = values[i + b];
          colidx_buf[threadIdx.y][i] = colidx[i + b];
        }
        __syncthreads();

        for (int k = 0; k < length; k++) {
          pares += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c];
          pares2 += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c2];
        }
      }
      if (rowptr[r + 1] - rowptr[r] > 0) {
        resMat[r][c] = pares;
        resMat[r][c2] = pares2;
      }
    }
  }
}

// no load balancing
__global__ void _spmm_cuda_v1_kernel_small(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> colidx,

    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int64_t nrows,
    const int64_t ncols,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> denseMat,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> resMat, int col_offset) {
  int r = blockIdx.x;
  int c = threadIdx.x + col_offset;
  if (c < denseMat.size(1)) {
    float pares = 0.0;
    for (int k = rowptr[r]; k < rowptr[r + 1]; k++) {
      pares += values[k] * denseMat[colidx[k]][c];
    }
    if (rowptr[r + 1] - rowptr[r] > 0) {
      resMat[r][c] += pares;
    }
  }
}

// no load balancing
torch::Tensor spmm_cuda_v1(torch::Tensor sparseMat, torch::Tensor denseMat) {
  auto rowidx = sparseMat.indices()[0].to(at::ScalarType::Int);
  auto colidx = sparseMat.indices()[1].to(at::ScalarType::Int);
  auto values = sparseMat.values();

  at::DeviceGuard g(denseMat.device());

  torch::Tensor resMat = torch::zeros({sparseMat.size(0), denseMat.size(1)}, denseMat.options());
  torch::Tensor rowptr = _to_csr_int(rowidx, sparseMat.size(0));

  cudaDeviceSynchronize();

  //std::cout << resMat.sizes() << std::endl;
  if (denseMat.size(1) >= 64) {
    dim3 nthreads_spmm(32, SROW_PER_TILE);
    dim3 nblocks_spmm(DIV(rowptr.size(0), SROW_PER_TILE), DIV(denseMat.size(1), 64));
    _spmm_cuda_v1_kernel<<<nblocks_spmm, nthreads_spmm>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    int remaining = denseMat.size(1) % 64;
    if (remaining > 0)
      _spmm_cuda_v1_kernel_small<<<rowptr.size(0) - 1, 64>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), denseMat.size(1) - remaining);

  } else {
    _spmm_cuda_v1_kernel_small<<<rowptr.size(0) - 1, 64>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), 0);
  }

  return resMat;
}

// load balancing
__global__ void _spmm_cuda_v2_kernel(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> vrowptr,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> colidx,

    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int64_t nrows,
    const int64_t ncols,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> denseMat,
    float *__restrict__ resMat) {
  __shared__ float values_buf[SROW_PER_TILE][LENGTH];
  __shared__ int colidx_buf[SROW_PER_TILE][LENGTH];

  int r = blockIdx.x * SROW_PER_TILE + threadIdx.y;
  if (r < vrowptr.size(0) - 1) {
    int c = blockIdx.y * 64 + threadIdx.x;
    int c2 = c + 32;

    if (c2 < ((denseMat.size(1) >> 6) << 6)) {
      float pares = 0.0;
      float pares2 = 0.0;

      for (int b = vrowptr[r]; b < vrowptr[r + 1]; b += LENGTH) {
        int length = LENGTH > vrowptr[r + 1] - b ? vrowptr[r + 1] - b : LENGTH;
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
          values_buf[threadIdx.y][i] = values[i + b];
          colidx_buf[threadIdx.y][i] = colidx[i + b];
        }
        __syncthreads();

        for (int k = 0; k < ((length >> 1) << 1); k += 2) {
          pares += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c];
          pares2 += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c2];
          pares += values_buf[threadIdx.y][k+1] * denseMat[colidx_buf[threadIdx.y][k + 1]][c];
          pares2 += values_buf[threadIdx.y][k+1] * denseMat[colidx_buf[threadIdx.y][k + 1]][c2];
        }

        if (length & 1) {
          pares += values_buf[threadIdx.y][length-1] * denseMat[colidx_buf[threadIdx.y][length - 1]][c];
          pares2 += values_buf[threadIdx.y][length-1] * denseMat[colidx_buf[threadIdx.y][length - 1]][c2];
        }
      }
      if (vrowptr[r + 1] - vrowptr[r] > 0) {
        int oidx = rowidx[vrowptr[r]] * denseMat.size(1) + c;
        atomicAdd(&resMat[oidx], pares);
        atomicAdd(&resMat[oidx + 32], pares2);
      }
    }
  }
}

// load balancing
__global__ void _spmm_cuda_v2_kernel_small(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> vrowptr,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> colidx,

    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int64_t nrows,
    const int64_t ncols,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> denseMat,
    float *__restrict__ resMat, int col_offset) {
  __shared__ float values_buf[SROW_PER_TILE][LENGTH];
  __shared__ int colidx_buf[SROW_PER_TILE][LENGTH];

  int r = blockIdx.x * SROW_PER_TILE + threadIdx.y;
  if (r < vrowptr.size(0) - 1) {
    int c = threadIdx.x + col_offset;
    float pares = 0.0;
    for (int b = vrowptr[r]; b < vrowptr[r + 1]; b += LENGTH) {
      int length = LENGTH > vrowptr[r + 1] - b ? vrowptr[r + 1] - b : LENGTH;
      for (int i = threadIdx.x; i < length; i += blockDim.x) {
        values_buf[threadIdx.y][i] = values[i + b];
        colidx_buf[threadIdx.y][i] = colidx[i + b];
      }
      __syncthreads();
      if (c < denseMat.size(1)) {
        for (int k = 0; k < ((length >> 1) << 1); k += 2) {
          pares += values_buf[threadIdx.y][k] * denseMat[colidx_buf[threadIdx.y][k]][c];
          pares += values_buf[threadIdx.y][k + 1] * denseMat[colidx_buf[threadIdx.y][k + 1]][c];
        }
        if (length & 1) {
          pares += values_buf[threadIdx.y][length-1] * denseMat[colidx_buf[threadIdx.y][length - 1]][c];
        }
      }
    }
    if (c < denseMat.size(1) && vrowptr[r + 1] - vrowptr[r] > 0) {
      atomicAdd(resMat + rowidx[vrowptr[r]] * denseMat.size(1) + c, pares);
    }
  }
}

__global__ void _calc_rowptr(torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr,
                             const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowidx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid <= rowidx.size(0)) {
    int pre = (tid == 0 ? -1 : rowidx[tid - 1]);
    int cur = (tid == rowidx.size(0) ? rowptr.size(0) - 1 : rowidx[tid]);
    for (int i = pre; i < cur; i++) {
      rowptr[i + 1] = tid;
    }
  }
}



__global__ void _calc_vrowptr(torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> vrowptr, const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowpos, const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr) {
  int r = threadIdx.x + blockIdx.x * blockDim.x;
  if (r < rowptr.size(0) - 1) {
    int j = 0;
    for (int i = rowptr[r]; i < rowptr[r + 1]; i += NNZ_PER_CHUNK) {
      vrowptr[rowpos[r] + j] = i;
      j++;
    }
  }
  if (r == rowptr.size(0)) {
    vrowptr[vrowptr.size(0) - 1] = rowptr[rowptr.size(0) - 1];
  }
}

__global__ void _prescan_arbitrary(int *__restrict__ output, int *__restrict__ input, int n, int powerOfTwo) {
  extern __shared__ int temp[];  // allocated on invocation

  int threadID = threadIdx.x;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  if (threadID < n) {
    temp[ai + bankOffsetA] = DIV(input[ai + 1] - input[ai], NNZ_PER_CHUNK);
    temp[bi + bankOffsetB] = DIV(input[bi + 1] - input[bi], NNZ_PER_CHUNK);
  } else {
    temp[ai + bankOffsetA] = 0;
    temp[bi + bankOffsetB] = 0;
  }

  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (threadID == 0) {
    temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;  // clear the last element
  }

  for (int d = 1; d < powerOfTwo; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (threadID < n) {
    output[ai] = temp[ai + bankOffsetA];
    output[bi] = temp[bi + bankOffsetB];
  }
}

__global__ void _prescan_arbitrary_real(int *__restrict__ output, int *__restrict__ input, int n, int powerOfTwo) {
  extern __shared__ int temp[];  // allocated on invocation

  int threadID = threadIdx.x;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  if (threadID < n) {
    temp[ai + bankOffsetA] = input[ai];
    temp[bi + bankOffsetB] = input[bi];
  } else {
    temp[ai + bankOffsetA] = 0;
    temp[bi + bankOffsetB] = 0;
  }

  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (threadID == 0) {
    temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;  // clear the last element
  }

  for (int d = 1; d < powerOfTwo; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (threadID < n) {
    output[ai] = temp[ai + bankOffsetA];
    output[bi] = temp[bi + bankOffsetB];
  }
}

__global__ void _prescan_large_real(int *output, int *input, int n, int *sums) {
  extern __shared__ int temp[];

  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = input[blockOffset + ai];
  temp[bi + bankOffsetB] = input[blockOffset + bi];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadID == 0) {
    sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[blockOffset + ai] = temp[ai + bankOffsetA];
  output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void _prescan_large(int *output, int *input, int n, int *sums) {
  extern __shared__ int temp[];

  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = DIV(input[blockOffset + ai + 1] - input[blockOffset + ai], NNZ_PER_CHUNK);
  temp[bi + bankOffsetB] = DIV(input[blockOffset + bi + 1] - input[blockOffset + bi], NNZ_PER_CHUNK);

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadID == 0) {
    sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[blockOffset + ai] = temp[ai + bankOffsetA];
  output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void _add(int *output, int length, int *n) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += n[blockID];
}

__global__ void _add(int *output, int length, int *n1, int *n2) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

__global__ void _add_not_real(int *output, int length, int *n1, int *n2) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += DIV(n1[blockID + 1] - n1[blockID], NNZ_PER_CHUNK) + n2[blockID];
}

// out stores the prefix sum, in stores the rowptr
// in is first used for calculating rowcount,
// then, rowcount is scanned to computed out
void scanSmallDeviceArray(int *out, int *in, int length, bool real) {
  int powerOfTwo = nextPowerOfTwo(length);
  if (real) {
    _prescan_arbitrary_real<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(out, in, length, powerOfTwo);
  } else {
    //std::cout << "here" << std::endl;
    _prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(out, in, length, powerOfTwo);
  }
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool real);

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool real) {
  int remainder = length % (ELEMENTS_PER_BLOCK);
  if (remainder == 0) {
    //std::cout << "length: " << length << std::endl;
    scanLargeEvenDeviceArray(d_out, d_in, length, real);
  } else {
    // perform a large scan on a compatible multiple of elements
    int lengthMultiple = length - remainder;
    //std::cout << lengthMultiple << std::endl;
    scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, real);

    // scan the remaining elements and add the (inclusive) last element of the large scan to this
    scanSmallDeviceArray(d_out + lengthMultiple, d_in + lengthMultiple, remainder, real);

    _add_not_real<<<1, remainder>>>(d_out + lengthMultiple, remainder, d_in + lengthMultiple - 1, d_out + lengthMultiple - 1);

    /* int *h_out = (int *)malloc(sizeof(int) * length);
    cudaMemcpy(h_out, d_out, sizeof(int) * length, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++) {
      std::cout << h_out[i] << std::endl;
    }*/
  }
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool real) {
  const int blocks = length / ELEMENTS_PER_BLOCK;
  const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

  int *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));

  if (real) {
    _prescan_large_real<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

  } else {
    _prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
  }

  const int sumsArrThreadsNeeded = (blocks + 1) / 2;
  // std::cout << "sumArr: " << sumsArrThreadsNeeded << std::endl;
  // int *h_sums = (int *)malloc(blocks * sizeof(int));
  // cudaMemcpy(h_sums, d_sums, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < blocks; i++) std::cout << h_sums[i] << " ";
  // std::cout << std::endl;
  if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
    // perform a large scan on the sums arr
    scanLargeDeviceArray(d_incr, d_sums, blocks, true);
  } else {
    // only need one block to scan sums arr so can use small scan
    scanSmallDeviceArray(d_incr, d_sums, blocks, true);
  }

  // int *h_incr = (int *)malloc(blocks * sizeof(int));
  // cudaMemcpy(h_incr, d_incr, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < blocks; i++) std::cout << h_incr[i] << " ";
  // std::cout << std::endl;

  _add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

  cudaFree(d_sums);
  cudaFree(d_incr);
}

// load balancing
torch::Tensor spmm_cuda_v2(torch::Tensor sparseMat, torch::Tensor denseMat) {
  auto rowidx = sparseMat.indices()[0].to(at::ScalarType::Int);
  auto colidx = sparseMat.indices()[1].to(at::ScalarType::Int);
  auto values = sparseMat.values();

  at::DeviceGuard g(denseMat.device());

  torch::Tensor resMat = torch::zeros({sparseMat.size(0), denseMat.size(1)}, denseMat.options());

  torch::Tensor rowptr = torch::empty({sparseMat.size(0) + 1}, rowidx.options());

  torch::Tensor rowcount = torch::zeros({sparseMat.size(0)}, rowidx.options());

  dim3 nthreads, nblocks;

  nthreads.x = 256;
  nblocks.x = DIV(rowidx.size(0) + 1, nthreads.x);

  _calc_rowptr<<<nblocks, nthreads>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

  CUDA_CALL(cudaDeviceSynchronize());

  // calcauted as prefix sum of rowptr divided by NNZ_PER_CHUNK, stores the writing positions for virtual rowptrs
  torch::Tensor rowpos = torch::zeros({sparseMat.size(0)}, rowidx.options());

  if (rowpos.size(0) > ELEMENTS_PER_BLOCK) {
    scanLargeDeviceArray(rowpos.data_ptr<int>(), rowptr.data_ptr<int>(), rowpos.size(0), false);

  } else {
    scanSmallDeviceArray(rowpos.data_ptr<int>(), rowptr.data_ptr<int>(), rowpos.size(0), false);
  }
  //std::cout << rowpos << std::endl;
  //std::cout << rowptr << std::endl;

  CUDA_CALL(cudaDeviceSynchronize());

  int num_virtual_rows = rowpos[rowpos.size(0) - 1].item().to<int>() + DIV(rowptr[rowptr.size(0) - 1].item().to<int>() - rowptr[rowptr.size(0) - 2].item().to<int>(), NNZ_PER_CHUNK);

  //std::cout << sparseMat.size(0) << " " << num_virtual_rows << std::endl;


  torch::Tensor vrowptr = torch::empty({num_virtual_rows + 1}, rowptr.options());

  nblocks.x = DIV(rowptr.size(0) + 1, nthreads.x);

  _calc_vrowptr<<<nblocks, nthreads>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowpos.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

  //std::cout << rowpos << std::endl;
  CUDA_CALL(cudaDeviceSynchronize());

  if (denseMat.size(1) >= 64) {
    //std::cout << vrowptr << std::endl;
    //std::cout << vrowptr << std::endl;
    dim3 nthreads_spmm(32, SROW_PER_TILE);
    dim3 nblocks_spmm(DIV(vrowptr.size(0) - 1, SROW_PER_TILE), DIV(denseMat.size(1), 64));
    _spmm_cuda_v2_kernel<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>());
    
    CUDA_CALL(cudaDeviceSynchronize());

    //std::cout << "55555" << std::endl;


    int remaining = denseMat.size(1) % 64;
    if (remaining > 0) {
      nthreads_spmm.x = 64;
      nblocks_spmm.y = 1;
      _spmm_cuda_v2_kernel_small<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>(), denseMat.size(1) - remaining);
    }
    CUDA_CALL(cudaDeviceSynchronize());
        //std::cout << "66666" << std::endl;
  } else {
         //   std::cout << "777777" << std::endl;

    dim3 nthreads_spmm(64, SROW_PER_TILE);
    dim3 nblocks_spmm(DIV(vrowptr.size(0) - 1, SROW_PER_TILE), 1);
    _spmm_cuda_v2_kernel_small<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), sparseMat.size(0), denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>(), 0);
    
    CUDA_CALL(cudaDeviceSynchronize());
          //  std::cout << "88888" << std::endl;

  }

  //cudaDeviceSynchronize();

  return resMat;
}

torch::Tensor spmm_cuda_v3(torch::Tensor rowidx, torch::Tensor colidx, torch::Tensor values, int nrows, torch::Tensor denseMat) {

  at::DeviceGuard g(denseMat.device());

  torch::Tensor resMat = torch::zeros({nrows, denseMat.size(1)}, denseMat.options());
  torch::Tensor rowptr = torch::empty({nrows + 1}, rowidx.options());
  torch::Tensor rowcount = torch::zeros({nrows}, rowidx.options());


  dim3 nthreads, nblocks;

  nthreads.x = 256;
  nblocks.x = DIV(rowidx.size(0) + 1, nthreads.x);


  _calc_rowptr<<<nblocks, nthreads>>>(rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

  cudaDeviceSynchronize();

  // calcauted as prefix sum of rowptr divided by NNZ_PER_CHUNK, stores the writing positions for virtual rowptrs
  torch::Tensor rowpos = torch::zeros({nrows}, rowidx.options());

  if (rowpos.size(0) > ELEMENTS_PER_BLOCK) {
    scanLargeDeviceArray(rowpos.data_ptr<int>(), rowptr.data_ptr<int>(), rowpos.size(0), false);

  } else {
    scanSmallDeviceArray(rowpos.data_ptr<int>(), rowptr.data_ptr<int>(), rowpos.size(0), false);
  }
  //std::cout << rowpos << std::endl;
  //std::cout << rowptr << std::endl;

  cudaDeviceSynchronize();

  int num_virtual_rows = rowpos[rowpos.size(0) - 1].item().to<int>() + DIV(rowptr[rowptr.size(0) - 1].item().to<int>() - rowptr[rowptr.size(0) - 2].item().to<int>(), NNZ_PER_CHUNK);

  //std::cout << sparseMat.size(0) << " " << num_virtual_rows << std::endl;

  torch::Tensor vrowptr = torch::empty({num_virtual_rows + 1}, rowptr.options());

  nblocks.x = DIV(rowptr.size(0) + 1, nthreads.x);

  _calc_vrowptr<<<nblocks, nthreads>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowpos.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

  cudaDeviceSynchronize();

  //std::cout << vrowptr << std::endl;
  //std::cout << rowptr << std::endl;
  //std::cout << rowidx << std::endl;

  //std::cout << "===================" << std::endl;

  //std::cout << rowidx.size(0) << std::endl;

  // std::cout << rowptr << std::endl;

  //std::cout << resMat << std::endl;

  if (denseMat.size(1) >= 64) {
    
    dim3 nthreads_spmm(32, SROW_PER_TILE);
    dim3 nblocks_spmm(DIV(vrowptr.size(0) - 1, SROW_PER_TILE), DIV(denseMat.size(1), 64));
    _spmm_cuda_v2_kernel<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), nrows, denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>());

    int remaining = denseMat.size(1) % 64;
    if (remaining > 0) {
      nthreads_spmm.x = 64;
      nblocks_spmm.y = 1;
      _spmm_cuda_v2_kernel_small<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), nrows, denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>(), denseMat.size(1) - remaining);
    }
  } else {
    dim3 nthreads_spmm(64, SROW_PER_TILE);
    dim3 nblocks_spmm(DIV(vrowptr.size(0) - 1, SROW_PER_TILE), 1);
    _spmm_cuda_v2_kernel_small<<<nblocks_spmm, nthreads_spmm>>>(vrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), nrows, denseMat.size(1), denseMat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), resMat.data_ptr<float>(), 0);
    ;
  }

  //cudaDeviceSynchronize();

  return resMat;
}

__global__ void _create_coo_tensor_kernel(
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> value,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> fullrowptr, 
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> rowptr,
    const torch::PackedTensorAccessor<int16_t, 1, torch::RestrictPtrTraits> colidx,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> normfact_row, 
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> normfact_col 
) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < rowptr.size(0) - 1) {
    for (int i = rowptr[tid]; i < rowptr[tid+1]; i++) {
      indices[0][i] = tid;
      indices[1][i] = colidx[i];
      value[i] = 1. / (fullrowptr[tid+1] - fullrowptr[tid]) * normfact_col[colidx[i]] * normfact_row[tid];
    }
  }
}


torch::Tensor to_coo_tensor(torch::Tensor fullrowptr, torch::Tensor rowptr, torch::Tensor colidx, torch::Tensor normfact_row, torch::Tensor normfact_col, int64_t nrows, int64_t ncols) {

    at::DeviceGuard g(colidx.device());

    auto options = colidx.options();
    options = options.dtype(torch::kLong);
    auto indices = torch::empty({2, colidx.size(0)}, options);
    options = options.dtype(torch::kFloat);
    auto value = torch::empty({colidx.size(0)}, options);

    dim3 nthreads, nblocks;

    nthreads.x = 256;
    nblocks.x = DIV(rowptr.size(0)-1, nthreads.x);

    _create_coo_tensor_kernel<<<nblocks, nthreads>>>(indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(), value.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), fullrowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), rowptr.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), colidx.packed_accessor<int16_t, 1, torch::RestrictPtrTraits>(), normfact_row.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), normfact_col.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

    CUDA_CALL(cudaDeviceSynchronize()); 

    return at::sparse_coo_tensor(indices, value, {nrows, ncols}).coalesce();

}
