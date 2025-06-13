#include <cuda_runtime.h>
#include <stdio.h>

#define N 256

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main() 
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of GPUs: %d\n", deviceCount);

    if (deviceCount < 2) {
        fprintf(stderr, "This example requires at least 2 GPUs.\n");
        return -1;
    }

    size_t dataSize = N * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_C0 = NULL, *d_C1 = NULL;

    // Allocate and initialize A and B on GPU 0
    cudaSetDevice(0);
    cudaMalloc((void**)&d_A, dataSize);
    cudaMalloc((void**)&d_B, dataSize);
    cudaMalloc((void**)&d_C0, dataSize);

    // Use init kernel to fill A and B
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C0, N);  // A and B are not initialized yet, but okay for structure

    // Manually initialize A and B
    float h_A[N], h_B[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMemcpy(d_A, h_A, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, dataSize, cudaMemcpyHostToDevice);

    // Compute vectorAdd on GPU 0
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C0, N);
    cudaDeviceSynchronize();

    // Allocate result buffer on GPU 1
    cudaSetDevice(1);
    cudaMalloc((void**)&d_C1, dataSize);


    // Copy result from GPU 0 to GPU 1
    cudaMemcpyPeerAsync(d_C1, 1, d_C0, 0, dataSize);
    cudaMemcpy(d_C1, d_C0, dataSize, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Copy to host and print
    float h_C[N];
    cudaMemcpy(h_C, d_C1, dataSize, cudaMemcpyDeviceToHost);

    printf("Sample result on GPU 1 (C = A + B): ");
    for (int i = 0; i < 5; ++i)
        printf("%f ", h_C[i]);
    printf("...\n");

    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C0);

    cudaSetDevice(1);
    cudaFree(d_C1);

    return 0;
}
