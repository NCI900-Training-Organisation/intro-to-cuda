#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono> // For timing the execution

// Kernel function: performs vector addition on the GPU
__global__ void vectorAdd2D(const float *A, const float *B, float *C, int N) 
{
    // Calculate global thread index using 2D block and grid coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert 2D index into a linear index for accessing 1D vectors
    int idx = row * gridDim.x * blockDim.x + col;

    // Perform element-wise addition if index is within bounds
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() 
{
    int N = 1 << 20; // Total number of elements (1 million)
    size_t size = N * sizeof(float); // Size in bytes for each vector

    // Allocate memory on the host (CPU)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host vectors with example data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Start timing using std::chrono (includes H2D, kernel, D2H)
    auto start = std::chrono::high_resolution_clock::now();

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

   
    // TODO: Adjust block size as needed and test the code for performance
    dim3 blockDim(16, 16);

    // Calculate how many total blocks are needed
    int totalThreads = blockDim.x * blockDim.y;
    int gridSize = (N + totalThreads - 1) / totalThreads;

    // Arrange grid as a square (approximately)
    dim3 gridDim((int)ceil(sqrt((float)gridSize)), (int)ceil(sqrt((float)gridSize)));

    // Launch the kernel on the GPU
    vectorAdd2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result back from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // End timing
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    // Verify correctness of results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    // Print result and total GPU time including memory copies
    printf("Vector addition with 2D threads %s!\n", success ? "succeeded" : "failed");
    printf("Total GPU time (H2D + kernel + D2H): %.4f ms\n", duration.count());

    // Free memory on both host and device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
