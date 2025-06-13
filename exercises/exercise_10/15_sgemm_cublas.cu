#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // Column-major index

void matrixMultiplyCuBLAS() 
{
    const int N = 2;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float h_A[N * N] = {1, 2, 3, 4};   // Row-major layout
    float h_B[N * N] = {1, 2, 3, 4};
    float h_C[N * N] = {0, 0, 0, 0};

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // C = A * B (row-major inputs treated as column-major)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  //TOD0: change to CUBLAS_OP_N
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result C = A x B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.1f ", h_C[IDX2C(i, j, N)]);
        }
        printf("\n");
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    matrixMultiplyCuBLAS();
    return 0;
}
