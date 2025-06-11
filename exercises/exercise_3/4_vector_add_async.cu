#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdlib.h>

#define NUM_STREAMS 4
#define NUM_ITERATIONS 10

__global__ void vectorAdd2D(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * gridDim.x * blockDim.x + col;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1 million elements per vector
    size_t size = N * sizeof(float);

    // Allocate host and device buffers per iteration
    float *h_A[NUM_ITERATIONS], *h_B[NUM_ITERATIONS], *h_C[NUM_ITERATIONS];
    float *d_A[NUM_ITERATIONS], *d_B[NUM_ITERATIONS], *d_C[NUM_ITERATIONS];
    cudaStream_t streams[NUM_STREAMS];

    // Create CUDA streams
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    // Allocate host memory (regular malloc) and device memory
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        h_A[i] = (float*)malloc(size);
        h_B[i] = (float*)malloc(size);
        h_C[i] = (float*)malloc(size);

        cudaMalloc(&d_A[i], size);
        cudaMalloc(&d_B[i], size);
        cudaMalloc(&d_C[i], size);
    }

    // Configure 2D block and grid
    dim3 blockDim(16, 16);
    int totalThreads = blockDim.x * blockDim.y;
    int gridSize = (N + totalThreads - 1) / totalThreads;
    dim3 gridDim((int)ceil(sqrt((float)gridSize)), (int)ceil(sqrt((float)gridSize)));

    // Start total timer
    auto start = std::chrono::high_resolution_clock::now();

    // Launch async vector addition operations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int stream_id = iter % NUM_STREAMS;

        // Initialize host data with unique values for this iteration
        for (int i = 0; i < N; ++i) {
            h_A[iter][i] = float(iter);
            h_B[iter][i] = float(i % 100);
        }

        // Asynchronous memory transfer and kernel execution
        cudaMemcpyAsync(d_A[iter], h_A[iter], size, cudaMemcpyHostToDevice, streams[stream_id]);
        cudaMemcpyAsync(d_B[iter], h_B[iter], size, cudaMemcpyHostToDevice, streams[stream_id]);

        vectorAdd2D<<<gridDim, blockDim, 0, streams[stream_id]>>>(
            d_A[iter], d_B[iter], d_C[iter], N
        );

        cudaMemcpyAsync(h_C[iter], d_C[iter], size, cudaMemcpyDeviceToHost, streams[stream_id]);
    }

    // Wait for all streams to finish
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
    }

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    printf("Total GPU time across %d iterations using %d streams: %.4f ms\n",
           NUM_ITERATIONS, NUM_STREAMS, duration.count());

    // Verify results
    bool allPassed = true;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (int i = 0; i < N; ++i) {
            float expected = float(iter) + float(i % 100);
            if (h_C[iter][i] != expected) {
                printf("Mismatch at iter %d, index %d: got %.1f, expected %.1f\n",
                       iter, i, h_C[iter][i], expected);
                allPassed = false;
                break;
            }
        }
    }

    printf("Result verification: %s\n", allPassed ? "PASSED" : "FAILED");

    // Cleanup
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
    }

    return 0;
}
