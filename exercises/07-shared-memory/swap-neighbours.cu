#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int threads = 4;

__global__ void swap_neighbours(int *output)
{
    __shared__ int values[threads];
    int t = threadIdx.x;

    values[t] = 10 * (t + 1);
    __syncthreads();

    // The last thread reads the first thread's slot.
    output[t] = values[(t + 1) % threads];
}

int main()
{
    int output[threads];
    int *device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output), sizeof(output)));

    swap_neighbours<<<1, threads>>>(device_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_output));

    std::puts("Thread  Wrote  Read");
    for (int t = 0; t < threads; ++t)
        std::printf("%6d  %5d  %4d\n", t, 10 * (t + 1), output[t]);
}
