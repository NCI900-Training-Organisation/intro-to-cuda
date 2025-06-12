#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: simple vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
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
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Begin CUDA Graph capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Copy inputs host->device (inside capture)
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // Kernel launch (recorded)
    vectorAdd<<<(N + 255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);

    // Copy result to host (recorded)
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // End capture
    cudaStreamEndCapture(stream, &graph);

    // Instantiate executable graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Execute the graph multiple times
    for (int i = 0; i < 5; ++i) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        printf("Run %d: C[100] = %f\n", i, h_C[100]);  // Should be 300 (100 + 200)
    }

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    return 0;
}
