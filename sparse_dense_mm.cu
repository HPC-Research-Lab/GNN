#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>

#define ALG CUSPARSE_SPMM_COO_ALG1


void CHECK_CUSPARSE_ERROR(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
      std::cerr << "ERROR" << std::endl;
        exit(1);
    }
}

void do_cusparse_spmm(
    int64_t a_rows, int64_t a_cols, int64_t a_nnz,
    float* a_values_dev,
    int64_t* a_row_indices_dev,
    int64_t* a_col_indices_dev,
    int64_t b_rows, int64_t b_cols, float *b_values_dev,
    int64_t c_rows, int64_t c_cols, float *c_values_dev
) {
    

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
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
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
        cudaMallocManaged(&buffer, bufferSize);
    }

    std::cout << "buffer size: " << bufferSize << std::endl;

    for (int i=0; i<2; i++) {
      std::cout << a_values_dev[i] << " ";
    }
    std::cout << std::endl;

    for(int i=0; i<9; i++) {
      std::cout << b_values_dev[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUSPARSE_ERROR(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
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

    if (bufferSize > 0) {
        cudaFree(buffer);
    }
    cusparseDestroySpMat(a_sparse_descr);
    cusparseDestroyDnMat(b_dense_descr);
    cusparseDestroyDnMat(c_dense_descr);
}


int main()
{
  // Create sparse matrix a
    int64_t a_rows = 3;
    int64_t a_cols = 3;
    int64_t a_nnz = 2;

    float* a_values_dev;
    cudaMallocManaged(&a_values_dev,a_nnz*sizeof(float));

    int64_t* a_row_indices_dev;
    cudaMallocManaged(&a_row_indices_dev,a_nnz*sizeof(int64_t));

    int64_t* a_col_indices_dev;
    cudaMallocManaged(&a_col_indices_dev,a_nnz*sizeof(int64_t));

    a_values_dev[0] = 1.0; a_values_dev[1] = 1.5;
    a_row_indices_dev[0] = 0; a_row_indices_dev[1] = 0;
    a_col_indices_dev[0] = 0; a_col_indices_dev[1] = 1;


    // Create dense matrix b
    int64_t b_rows = 3;
    int64_t b_cols = 3;
    float* b_values_dev;
    cudaMallocManaged(&b_values_dev, b_rows*b_cols*sizeof(float));
    for (int i=0; i<9; i++) b_values_dev[i] = (float)i;

    // Create matrix c
    int64_t c_rows = 3;
    int64_t c_cols = 3;
    float* c_values_dev;
    cudaMallocManaged(&c_values_dev, c_rows*c_cols*sizeof(float)); 

    do_cusparse_spmm(a_rows, a_cols, a_nnz, a_values_dev, a_row_indices_dev, a_col_indices_dev, b_rows, b_cols, b_values_dev, c_rows, c_cols, c_values_dev);



    for(int i=0; i<9; i++) {
      std::cout << c_values_dev[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


