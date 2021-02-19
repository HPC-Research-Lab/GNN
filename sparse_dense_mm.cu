#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>

#define ALG CUSPARSE_SPMM_COO_ALG1


extern "C" void do_cusparse_spmm(
    int64_t a_rows, int64_t a_cols, int64_t a_nnz,
    void* a_values_dev,
    void* a_row_indices_dev,
    void* a_col_indices_dev,
    int64_t b_rows, int64_t b_cols, void *b_values_dev,
    int64_t c_rows, int64_t c_cols, void *c_values_dev, bool trans_a
);


void CHECK_CUSPARSE_ERROR(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
      if (status == CUSPARSE_STATUS_EXECUTION_FAILED) {
        std::cerr << "execution failed" << std::endl;
      } else if (status == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES	
) {
  std::cerr << "insufficient resource" << std::endl;
} else if (status == CUSPARSE_STATUS_INTERNAL_ERROR	
) {
  std::cerr << "internal error" << std::endl;
}
      std::cerr << "ERROR: " << std::endl;
        exit(1);
    }
}


void do_cusparse_spmm(
    int64_t a_rows, int64_t a_cols, int64_t a_nnz,
    void* a_values_dev_a,
    void* a_row_indices_dev_a,
    void* a_col_indices_dev_a,
    int64_t b_rows, int64_t b_cols, void *b_values_dev_a,
    int64_t c_rows, int64_t c_cols, void *c_values_dev_a, 
    bool trans_a
) {
    
    float *a_values_dev = (float *)a_values_dev_a;
    float *b_values_dev = (float *)b_values_dev_a;
    float *c_values_dev = (float *)c_values_dev_a;
    int32_t *a_row_indices_dev = (int32_t*)a_row_indices_dev_a;
    int32_t *a_col_indices_dev = (int32_t*)a_col_indices_dev_a;

    cudaSetDevice(3);

    cusparseOperation_t transa_flag = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transb_flag = CUSPARSE_OPERATION_TRANSPOSE;
    if (trans_a) {
      transa_flag = CUSPARSE_OPERATION_TRANSPOSE;
      transb_flag = CUSPARSE_OPERATION_NON_TRANSPOSE;
    }

    cusparseSpMatDescr_t a_sparse_descr;
    CHECK_CUSPARSE_ERROR(cusparseCreateCoo(
        &a_sparse_descr,
        a_rows,
        a_cols,
        a_nnz,
        a_row_indices_dev,
        a_col_indices_dev,
        a_values_dev,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));

    cusparseDnMatDescr_t b_dense_descr;
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(
        &b_dense_descr,
        b_rows,
        b_cols,
        b_rows,
        b_values_dev,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL
    ));

    cusparseDnMatDescr_t c_dense_descr;
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(
        &c_dense_descr,
        c_rows,
        c_cols,
        c_rows,
        c_values_dev,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL
    ));


    float alpha = 1;
    float beta = 0;

    cusparseHandle_t handle;
    CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    size_t bufferSize;

    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(
        handle,
        transa_flag,
        transb_flag,
        (void*)&alpha,
        a_sparse_descr,
        b_dense_descr,
        (void*)&beta,
        c_dense_descr,
        CUDA_R_32F,
        ALG,
        &bufferSize
    ));
    cudaDeviceSynchronize();

    void* buffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }
   // std::cout << "sssssss" << std::endl;

//    int32_t *a = (int32_t*)malloc(sizeof(int32_t)*a_nnz);
  //  cudaMemcpy(a, a_row_indices_dev, sizeof(int32_t) * a_nnz, cudaMemcpyDeviceToHost);
    //std::cout << a[0] << " " << a[1] << " " << a[a_nnz-1] << std::endl;

//      int32_t *ac = (int32_t*)malloc(sizeof(int32_t)*a_nnz);
  //  cudaMemcpy(ac, a_col_indices_dev, sizeof(int32_t) * a_nnz, cudaMemcpyDeviceToHost);
    //std::cout << ac[0] << " " << ac[1] << " " << ac[a_nnz-1] << std::endl;
    //std::cout << "hhhhhhhhhh" << std::endl;



  //  float *b = (float*)malloc(sizeof(float) * b_rows * b_cols);
 //   cudaMemcpy(b, b_values_dev, sizeof(float) * b_rows * b_cols, cudaMemcpyDeviceToHost);
 //       std::cout << b[0] << " " << b[1] << " " << b[b_rows*b_cols-1] << std::endl;

    //std::cout << "********!!!!!" << std::endl;

   // float *c = (float*)malloc(sizeof(float) * c_rows * c_cols);
   // cudaMemcpy(c, c_values_dev, sizeof(float) * c_rows * c_cols, cudaMemcpyDeviceToHost);
     //   std::cout << c[0] << " " << c[1] << " " << c[c_rows*c_cols-1] << std::endl;

    CHECK_CUSPARSE_ERROR(cusparseSpMM(
        handle,
        transa_flag,
        transb_flag,
        (void*)&alpha,
        a_sparse_descr,
        b_dense_descr,
        (void*)&beta,
        c_dense_descr,
        CUDA_R_32F,
        ALG,
        buffer
    ));


    cudaDeviceSynchronize();
//std::cout << "xxxxx" << std::endl;
//cudaMemcpy(c, c_values_dev, sizeof(float) * c_rows * c_cols, cudaMemcpyDeviceToHost);
  //      for (int i=0; i<c_rows*c_cols; i++) std::cout << c[i] << std::endl;

    if (bufferSize > 0) {
        cudaFree(buffer);
    }
    cusparseDestroySpMat(a_sparse_descr);
    cusparseDestroyDnMat(b_dense_descr);
    cusparseDestroyDnMat(c_dense_descr);
}

/*
int main()
{
  // Create sparse matrix a
    int64_t a_rows = 2;
    int64_t a_cols = 3;
    int64_t a_nnz = 6;

    float* a_values_dev;
    cudaMallocManaged(&a_values_dev,a_nnz*sizeof(float));

    int32_t* a_row_indices_dev;
    cudaMallocManaged(&a_row_indices_dev,a_nnz*sizeof(int64_t));

    int32_t* a_col_indices_dev;
    cudaMallocManaged(&a_col_indices_dev,a_nnz*sizeof(int64_t));

    a_values_dev[0] = 1.0; a_values_dev[1] = 1; a_values_dev[2] = 1.0; a_values_dev[3] = 1; a_values_dev[4] = 1.0; a_values_dev[5] = 1;
    a_row_indices_dev[0] = 0; a_row_indices_dev[1] = 0; a_row_indices_dev[2] = 0; a_row_indices_dev[3] = 1; a_row_indices_dev[4] = 1; a_row_indices_dev[5] = 1;
    a_col_indices_dev[0] = 0; a_col_indices_dev[1] = 1; a_col_indices_dev[2] = 2; a_col_indices_dev[3] = 0;
    a_col_indices_dev[4] = 1; a_col_indices_dev[5] = 2;


    // Create dense matrix b
    int64_t b_rows = 2;
    int64_t b_cols = 3;
    float* b_values_dev;
    cudaMallocManaged(&b_values_dev, b_rows*b_cols*sizeof(float));
    for (int i=0; i<6; i++) b_values_dev[i] = 1.0;

    // Create matrix c
    int64_t c_rows = 2;
    int64_t c_cols = 2;
    float* c_values_dev;
    cudaMallocManaged(&c_values_dev, c_rows*c_cols*sizeof(float)); 

    do_cusparse_spmm(a_rows, a_cols, a_nnz, a_values_dev, a_row_indices_dev, a_col_indices_dev, b_rows, b_cols, b_values_dev, c_rows, c_cols, c_values_dev);



    for(int i=0; i<4; i++) {
      std::cout << c_values_dev[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
*/

