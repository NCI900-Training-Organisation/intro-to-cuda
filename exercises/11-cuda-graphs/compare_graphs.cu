#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void checkCuda(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() 
{
    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int iterations = 100;

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size), "Alloc d_A");
    checkCuda(cudaMalloc(&d_B, size), "Alloc d_B");
    checkCuda(cudaMalloc(&d_C, size), "Alloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy h_A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy h_B");

    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Traditional execution ---
    checkCuda(cudaEventRecord(start, stream), "Start traditional");
    for (int i = 0; i < iterations; ++i) {
        vectorAdd<<<(N + 255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
        cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    }
    checkCuda(cudaEventRecord(stop, stream), "Stop traditional");
    cudaEventSynchronize(stop);
    float time_traditional = 0.0f;
    cudaEventElapsedTime(&time_traditional, start, stop);
    printf("[Traditional] Time: %.3f ms (avg %.3f ms/iteration)\n",
           time_traditional, time_traditional / iterations);

    // --- CUDA Graph execution ---
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    vectorAdd<<<(N + 255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    checkCuda(cudaEventRecord(start, stream), "Start graph");
    for (int i = 0; i < iterations; ++i) {
        cudaGraphLaunch(graphExec, stream);
    }
    checkCuda(cudaEventRecord(stop, stream), "Stop graph");
    cudaEventSynchronize(stop);
    float time_graph = 0.0f;
    cudaEventElapsedTime(&time_graph, start, stop);
    printf("[Graph]       Time: %.3f ms (avg %.3f ms/iteration)\n",
           time_graph, time_graph / iterations);

    // --- Cleanup ---
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}
