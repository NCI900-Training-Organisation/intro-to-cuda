// Launch two blocks with four threads per block and inspect CUDA's built-in indices.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void print_launch_info()
{
    printf("Thread index: %u | Block index: %u | Block dim: %u | Grid dim: %u\n",
           threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}

int main()
{
    print_launch_info<<<2, 4>>>();

    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        // Wait for every thread to finish and flush the kernel's printf output.
        error = cudaDeviceSynchronize();
    }
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
