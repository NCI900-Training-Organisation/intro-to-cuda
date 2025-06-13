#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

#define NUM_STREAMS 4
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

    float *A[NUM_ITERATIONS], *B[NUM_ITERATIONS], *C[NUM_ITERATIONS];
    cudaStream_t streams[NUM_STREAMS];

    // Create streams
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    // Allocate unified memory
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaMallocManaged(&A[i], size);
        cudaMallocManaged(&B[i], size);
        cudaMallocManaged(&C[i], size);
    }

    // Configure block and grid
    dim3 blockDim(16, 16);
    int totalThreads = blockDim.x * blockDim.y;
    int gridSize = (N + totalThreads - 1) / totalThreads;
    dim3 gridDim((int)ceil(sqrt((float)gridSize)), (int)ceil(sqrt((float)gridSize)));

    // Create overall timing events
    cudaEvent_t overallStart, overallStop;
    cudaEventCreate(&overallStart);
    cudaEventCreate(&overallStop);

    // Start overall timer
    cudaEventRecord(overallStart);

    // Launch kernels asynchronously
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int stream_id = iter % NUM_STREAMS;

        // Initialize inputs
        for (int i = 0; i < N; ++i) {
            A[iter][i] = float(iter);
            B[iter][i] = float(i % 100);
        }

        vectorAdd2D<<<gridDim, blockDim, 0, streams[stream_id]>>>(
            A[iter], B[iter], C[iter], N);

    }

    // Wait for all streams to finish
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
    }

    // Stop overall timer
    cudaEventRecord(overallStop);
    cudaEventSynchronize(overallStop);

    float totalTime = 0;
    cudaEventElapsedTime(&totalTime, overallStart, overallStop);
    printf("Total time for %d iterations using %d streams: %.4f ms\n", NUM_ITERATIONS, NUM_STREAMS, totalTime);

    // Verify results
    bool allPassed = true;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        for (int i = 0; i < N; ++i) {
            float expected = float(iter) + float(i % 100);
            if (C[iter][i] != expected) {
                printf("Mismatch at iter %d, index %d: got %.1f, expected %.1f\n",
                       iter, i, C[iter][i], expected);
                allPassed = false;
                break;
            }
        }
    }

    printf("Result verification: %s\n", allPassed ? "PASSED" : "FAILED");

    // Cleanup
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
    }

    cudaEventDestroy(overallStart);
    cudaEventDestroy(overallStop);

    return 0;
}
