#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_ITERATIONS 10

__global__ void vectorAdd2D(const float *A, const float *B, float *C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * gridDim.x * blockDim.x + col;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() 
{
    const int N = 1 << 20; // 1 million elements per vector
    size_t size = N * sizeof(float);

    // Allocate host memory (reused across iterations)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Allocate device memory (reused across iterations)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Configure 2D thread/block setup once
    dim3 blockDim(16, 16);
    int totalThreads = blockDim.x * blockDim.y;
    int gridSize = (N + totalThreads - 1) / totalThreads;
    dim3 gridDim((int)ceil(sqrt((float)gridSize)), (int)ceil(sqrt((float)gridSize)));

    // Start timing full loop
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Initialize A and B with different values for each iteration
        for (int i = 0; i < N; ++i) {
            h_A[i] = float(iter);        // A is filled with the current iteration number
            h_B[i] = float(i % 100);     // B is filled with a pattern (0 to 99 repeated)
        }

        // Copy host data to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Launch kernel
        vectorAdd2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

        // Copy result back to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Verify result (optional)
        bool success = true;
        for (int i = 0; i < N; ++i) {
            float expected = float(iter) + float(i % 100);
            if (h_C[i] != expected) {
                success = false;
                printf("Mismatch at iter %d, index %d: expected %.1f, got %.1f\n",
                       iter, i, expected, h_C[i]);
                break;
            }
        }

        printf("Iteration %d: Vector addition %s\n", iter, success ? "succeeded" : "failed");
    }

    // Stop total timing
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalDuration = stop - start;

    printf("Total GPU time for %d vector additions: %.4f ms\n", NUM_ITERATIONS, totalDuration.count());

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
