// Compare the block-local thread index with a unique global index.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void print_global_index()
{
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread index: %u | Block index: %u | Block dim: %u | Grid dim: %u | Global index: %u\n",
           threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, global_index);
}

int main()
{
    print_global_index<<<2, 4>>>();

    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
