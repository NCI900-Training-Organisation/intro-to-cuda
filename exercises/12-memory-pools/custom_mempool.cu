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
    printf("  Reserved memory:      %lu bytes (%.2f MB)\n", current, current / (1024.0 * 1024));
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

    // Create a custom memory pool for device 0
    cudaMemPool_t myPool;
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned; // Pinned host memory
    props.handleTypes = cudaMemHandleTypeNone;     // No interprocess sharing
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;                          // GPU device 0

    cudaMemPoolCreate(&myPool, &props);

    // Configure the custom memory pool attributes
    int threshold = 1024 * 1024; // 1 MB threshold for releasing memory
    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReleaseThreshold, &threshold);          // 1 MB threshold
    int current = 512 * 1024 * 1024;
    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReservedMemCurrent, &current);  // 512 MB reserved (informational)
    int high = 1024 * 1024 * 1024; // 1 GB high limit
    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReservedMemHigh, &high);    // 1 GB high limit (informational)
    //cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReusePolicy, cudaMemPoolReusePolicyAggressive);  // Aggressive reuse
    //cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrAllocationGranularity, 256 * 1024);      // 256 KB granularity

    // Show pool usage before allocation
    printPoolUsage(myPool, "Before Allocation");

    // Allocate device memory using the custom memory pool asynchronously
    cudaMallocFromPoolAsync((void**)&d_A, size, myPool, stream);
    cudaMallocFromPoolAsync((void**)&d_B, size, myPool, stream);
    cudaMallocFromPoolAsync((void**)&d_C, size, myPool, stream);

    // Wait for allocations to complete
    cudaStreamSynchronize(stream);

    // Show pool usage after allocation
    printPoolUsage(myPool, "After Allocation");

    // Copy data from host to device asynchronously
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // Launch kernel
    vectorAdd<<<(N + 255) / 256, 256, 0, stream>>>(d_A, d_B, d_C, N);

    // Copy result back to host asynchronously
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations to finish
    cudaStreamSynchronize(stream);

    printf("C[100] = %f\n", h_C[100]);  // Should be 100 + 2*100 = 300

    // Free device memory asynchronously
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);

    // Synchronize before trimming
    cudaStreamSynchronize(stream);

    // Trim the pool to release unused memory back to the system
    cudaMemPoolTrimTo(myPool, 0);  // Release all unused memory

    // Show pool usage after freeing and trimming
    printPoolUsage(myPool, "After Free and Trim");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    // Destroy the custom memory pool
    cudaMemPoolDestroy(myPool);

    return 0;
}
