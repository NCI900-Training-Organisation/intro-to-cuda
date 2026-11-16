#include <stdio.h>
#include <cuda_runtime.h>

// Child kernel (launched from the device)
__global__ void childKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

// Parent kernel (launched from the host, and launches child from device)
__global__ void parentKernel(int *data, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Launch child kernel from the device (this is dynamic parallelism!)
    childKernel<<<blocks, threads>>>(data, n);
    
    //TODO: // Uncomment the following line to synchronize after launching the child kernel
    //cudaDeviceSynchronize();
}

int main()
{
    const int N = 1024;
    size_t size = N * sizeof(int);

    int *h_data = (int *)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    int *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch parent kernel
    parentKernel<<<1, 1>>>(d_data, N);

    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    cudaFree(d_data);
    free(h_data);
    return 0;
}
