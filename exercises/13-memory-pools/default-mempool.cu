#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: simple vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void printPoolUsage(cudaMemPool_t pool, const char* label) {
    size_t current, high;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &current);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high);

    printf("%s:\n", label);
    printf("  Reserved memory:     %lu bytes (%.2f MB)\n", current, current / (1024.0 * 1024));
    printf("  Peak reserved memory: %lu bytes (%.2f MB)\n\n", high, high / (1024.0 * 1024));
}

int main() 
{
    const int N = 1 << 16;
    const int size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host input arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    float *d_A, *d_B, *d_C;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Get the default memory pool for device 0
    cudaMemPool_t defaultPool;
    cudaDeviceGetDefaultMemPool(&defaultPool, 0);

    // Show pool usage before allocation
    printPoolUsage(defaultPool, "Before Allocation");

    // Allocate device memory using default memory pool
    cudaMallocAsync((void**)&d_A, size, stream);
    cudaMallocAsync((void**)&d_B, size, stream);
    cudaMallocAsync((void**)&d_C, size, stream);

    // Wait for allocation to complete
    cudaStreamSynchronize(stream);

    // Show pool usage after allocation
    printPoolUsage(defaultPool, "After Allocation");

    // Asynchronous memory copy to device
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // Launch kernel
    vectorAdd<<<(N + 255) / 256, 256, 0, stream>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations to finish
    cudaStreamSynchronize(stream);

    printf("C[100] = %f\n", h_C[100]);  // Should be 100 + 2*100 = 300

    // Free device memory asynchronously
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);

    // Final sync before cleanup
    cudaStreamSynchronize(stream);

    // Show pool usage after deallocation (memory is returned to the pool, not system)
    printPoolUsage(defaultPool, "After Free (Pool still holds memory)");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    return 0;
}
