#include <stdio.h>
#include <cuda_runtime.h> // Provides access to CUDA runtime API functions


__global__ void add_vectors(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    int n = 1024;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(n * sizeof(float));
    h_b = (float*)malloc(n * sizeof(float));
    h_c = (float*)malloc(n * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block and enough blocks to cover all elements
    add_vectors<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}