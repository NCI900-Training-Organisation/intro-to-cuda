// One thread records one pixel's timestamp. One block keeps all clocks on one SM.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void record_pixels(unsigned long long *timestamps)
{
    // Every thread reaches this barrier. It does not impose warp execution order.
    __syncthreads();
    const unsigned long long timestamp = clock64();
    timestamps[threadIdx.x] = timestamp;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s output.csv\n", argv[0]);
        return EXIT_FAILURE;
    }
    constexpr int threads = 256;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::fprintf(stderr, "GPU: %s; one block, %d threads, %d warps\n",
                 properties.name, threads, threads / properties.warpSize);
    unsigned long long *device_timestamps = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_timestamps),
                          threads * sizeof(unsigned long long)));
    record_pixels<<<1, threads>>>(device_timestamps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long timestamps[threads];
    CUDA_CHECK(cudaMemcpy(timestamps, device_timestamps, sizeof(timestamps),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_timestamps));
    FILE *output = std::fopen(argv[1], "w");
    if (!output) {
        std::perror(argv[1]);
        return EXIT_FAILURE;
    }
    std::fprintf(output, "thread,warp,lane,clock\n");
    for (int thread = 0; thread < threads; ++thread) {
        std::fprintf(output, "%d,%d,%d,%llu\n", thread,
                     thread / properties.warpSize, thread % properties.warpSize,
                     timestamps[thread]);
    }
    if (std::fclose(output) != 0) {
        std::perror("Closing output CSV");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
