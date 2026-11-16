#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define NUM_ITERATIONS 10

// Simple macro to check CUDA API results
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                          \
} while(0)

// Simplified kernel: no shared memory, direct vector add
__global__ void vectorAdd2D(const float *A, const float *B, float *C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute linear index from 2D indices
    int width = gridDim.x * blockDim.x;  // total threads in x dimension
    int idx = row * width + col;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() 
{
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *h_A[NUM_ITERATIONS], *h_B[NUM_ITERATIONS], *h_C[NUM_ITERATIONS];
    float *d_A[NUM_ITERATIONS], *d_B[NUM_ITERATIONS], *d_C[NUM_ITERATIONS];
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t startEvents[NUM_ITERATIONS], stopEvents[NUM_ITERATIONS];

    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        h_A[i] = (float *)malloc(size);
        h_B[i] = (float *)malloc(size);
        h_C[i] = (float *)malloc(size);

        CUDA_CHECK(cudaMalloc((void **)&d_A[i], size));
        CUDA_CHECK(cudaMalloc((void **)&d_B[i], size));
        CUDA_CHECK(cudaMalloc((void **)&d_C[i], size));

        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
    }

    // Get occupancy-based block size (1D for simplicity)
    int minGridSize = 0, blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,  // minimum grid size needed to achieve the best potential
        &blockSize,    // Block size
        vectorAdd2D,   // Kernel function
        0,             // Per-block dynamic shared memory usage intended, in bytes
        0));           //  The maximum block size func is designed to work with. 0 means no limit.

    // Convert 1D blockSize into 2D blockDim (square)
    int side = (int) sqrtf((float)blockSize);
    if (side == 0) side = 16;  // fallback
    dim3 blockDim(side, side);

    int totalThreads = blockDim.x * blockDim.y;
    int gridSize = (N + totalThreads - 1) / totalThreads;
    int gridSide = (int)ceil(sqrtf((float)gridSize));
    dim3 gridDim(gridSide, gridSide);

    int activeBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        vectorAdd2D,
        totalThreads,
        0));

    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

    printf("Chosen blockDim = (%d, %d) => %d threads\n", blockDim.x, blockDim.y, totalThreads);
    printf("GridDim = (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Max active blocks per SM: %d\n", activeBlocksPerSM);
    printf("Total SMs: %d\n", numSMs);
    printf("Estimated max active blocks on device: %d\n\n", activeBlocksPerSM * numSMs);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        int stream_id = iter % NUM_STREAMS;

        // Initialize host arrays
        for (int i = 0; i < N; ++i) {
            h_A[iter][i] = (float)iter;
            h_B[iter][i] = (float)(i % 100);
        }

        CUDA_CHECK(cudaEventRecord(startEvents[iter], streams[stream_id]));

        CUDA_CHECK(cudaMemcpyAsync(d_A[iter], h_A[iter], size, cudaMemcpyHostToDevice, streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(d_B[iter], h_B[iter], size, cudaMemcpyHostToDevice, streams[stream_id]));

        vectorAdd2D<<<gridDim, blockDim, 0, streams[stream_id]>>>(
            d_A[iter], d_B[iter], d_C[iter], N);

        CUDA_CHECK(cudaMemcpyAsync(h_C[iter], d_C[iter], size, cudaMemcpyDeviceToHost, streams[stream_id]));

        CUDA_CHECK(cudaEventRecord(stopEvents[iter], streams[stream_id]));
    }

    // Wait for all streams to finish
    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    // Get and print timing results
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        CUDA_CHECK(cudaEventSynchronize(stopEvents[iter]));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvents[iter], stopEvents[iter]));
        printf("Iteration %d took %.3f ms\n", iter, ms);
    }

    // Cleanup
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }

    for (int s = 0; s < NUM_STREAMS; ++s)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));

    return 0;
}
