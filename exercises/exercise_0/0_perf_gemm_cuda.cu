#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * Macro to check CUDA runtime errors
 */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * Map cuBLAS status codes to human-readable strings
 */
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

/**
 * Macro to check cuBLAS errors
 */
#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cublasGetErrorString(status));        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    int N = 1000;        // Matrix dimension (N x N)
    int runs = 5;        // Number of GEMM runs

    // -------------------------------
    // 1. Create cuBLAS handle
    // -------------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // -------------------------------
    // 2. Set GPU device
    // -------------------------------
    int32_t gpuId = 0; 
    CHECK_CUDA(cudaSetDevice(gpuId));

    // -------------------------------
    // 3. Create CUDA stream
    // -------------------------------
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // -------------------------------
    // 4. Allocate device memory
    // -------------------------------

     // Allocate matrices on CPU
    double *hostPtrA = (double *)malloc(sizeof(double) * N * N);
    double *hostPtrB = (double *)malloc(sizeof(double) * N * N);
    double *hostPtrC = (double *)malloc(sizeof(double) * N * N);

    double *devPtrA, *devPtrB, *devPtrC;
    CHECK_CUDA(cudaMalloc((double **)&devPtrA, sizeof(double) * N * N));
    CHECK_CUDA(cudaMalloc((double **)&devPtrB, sizeof(double) * N * N));
    CHECK_CUDA(cudaMalloc((double **)&devPtrC, sizeof(double) * N * N));

    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        hostPtrA[i] = 1.0;
        hostPtrB[i] = 2.0;
        hostPtrC[i] = 0.0;
    }

    CHECK_CUDA(cudaMemcpy(devPtrA, hostPtrA, sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(devPtrB, hostPtrB, sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(devPtrC, hostPtrC, sizeof(double) * N * N, cudaMemcpyHostToDevice));

    // -------------------------------
    // 5. Set GEMM scalars
    // -------------------------------
    double alpha = 1.0;
    double beta  = 1.0;

    // -------------------------------
    // 6. Create CUDA events for timing
    // -------------------------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // -------------------------------
    // 7. Launch GEMM 'runs' times
    // -------------------------------
    for (int i = 0; i < runs; i++) {
        // Record start event
        CHECK_CUDA(cudaEventRecord(start, stream));

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        CHECK_CUBLAS(cublasDgemm(handle, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            N, 
            N, 
            N, 
            &alpha,
            devPtrA, N,
            devPtrB, N,
            &beta,
            devPtrC, N));

        // Record stop event and synchronize
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        // Compute elapsed time in milliseconds
        float elapsed_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        printf("Run %d completed in %.3f ms.\n", i + 1, elapsed_ms);
    }

    // -------------------------------
    // 8. Cleanup
    // -------------------------------
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(devPtrA));
    CHECK_CUDA(cudaFree(devPtrB));
    CHECK_CUDA(cudaFree(devPtrC));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
