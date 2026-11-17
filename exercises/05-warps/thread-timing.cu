// One block of 1,024 threads: record one clock reading per thread on the same SM.
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

__global__ void record_thread_timestamps(unsigned long long *timestamps)
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
    constexpr int threads = 1024;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0]) {
        std::fprintf(stderr, "This GPU cannot launch a one-dimensional block of %d threads.\n", threads);
        return EXIT_FAILURE;
    }
    const int warps = (threads + properties.warpSize - 1) / properties.warpSize;
    const int last_warp_threads = threads - (warps - 1) * properties.warpSize;
    std::fprintf(stderr, "GPU: %s; one block, %d threads, %d warps\n",
                 properties.name, threads, warps);
    std::fprintf(stderr, "The final warp contains %d active threads.\n", last_warp_threads);
    unsigned long long *device_timestamps = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_timestamps),
                          threads * sizeof(unsigned long long)));
    // One block contains 32 full warps on a GPU with 32-thread warps.
    // Every thread participates in the barrier before taking its clock reading.
    record_thread_timestamps<<<1, threads>>>(device_timestamps);
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
    std::printf("Saved %d thread clock readings to %s. Plot them in 05-warps.ipynb.\n", threads, argv[1]);
    return EXIT_SUCCESS;
}
