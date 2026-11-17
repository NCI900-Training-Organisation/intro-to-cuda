// Compute image brightness with a shared-memory reduction, then adjust every pixel.
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void sum_pixels(const unsigned char *pixels, unsigned long long *block_sums, int count)
{
    extern __shared__ unsigned long long shared[];
    int local = threadIdx.x;
    int global = blockIdx.x * blockDim.x + local;
    // Incomplete final blocks contribute zeros. Every thread still reaches every barrier.
    shared[local] = global < count ? pixels[global] : 0;
    __syncthreads();

    // This reduction requires a power-of-two block size.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local < stride)
            shared[local] += shared[local + stride];
        __syncthreads();
    }
    if (local == 0)
        block_sums[blockIdx.x] = shared[0];
}

__global__ void adjust_pixels(const unsigned char *input, unsigned char *output,
                              int count, double gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double value = input[i] * gain;
        output[i] = static_cast<unsigned char>(fmin(255.0, floor(value + 0.5)));
    }
}

int integer(const char *text, int maximum)
{
    char *end;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > maximum) {
        std::fprintf(stderr, "Expected integer in [1, %d]: %s\n", maximum, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int main(int argc, char **argv)
{
    if (argc != 9) {
        std::fprintf(stderr, "Usage: %s input.raw width height target threads output.raw stats.json block-sums.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    int width = integer(argv[2], 4096), height = integer(argv[3], 4096);
    int target = integer(argv[4], 255), threads = integer(argv[5], 1024);
    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if ((threads & (threads - 1)) || threads > device.maxThreadsPerBlock) {
        std::fprintf(stderr, "Threads must be a power of two supported by this GPU.\n");
        return EXIT_FAILURE;
    }
    int count = width * height, blocks = (count + threads - 1) / threads;
    std::vector<unsigned char> input(count), output(count);
    std::ifstream source(argv[1], std::ios::binary);
    if (!source.read(reinterpret_cast<char *>(input.data()), count) || source.peek() != EOF) {
        std::fprintf(stderr, "Input must contain exactly width * height grayscale bytes.\n");
        return EXIT_FAILURE;
    }
    unsigned char *device_input = nullptr, *device-output = nullptr;
    unsigned long long *device_sums = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_input), count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device-output), count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_sums), blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), count, cudaMemcpyHostToDevice));

    sum_pixels<<<blocks, threads, threads * sizeof(unsigned long long)>>>(device_input, device_sums, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> sums(blocks);
    CUDA_CHECK(cudaMemcpy(sums.data(), device_sums, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    // Blocks cannot share shared memory. For this teaching example, the CPU combines block totals.
    unsigned long long total = 0;
    for (auto sum : sums) total += sum;
    double mean = static_cast<double>(total) / count;
    double gain = mean > 0.0 ? target / mean : 1.0;
    adjust_pixels<<<blocks, threads>>>(device_input, device-output, count, gain);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device-output, count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device-output));
    CUDA_CHECK(cudaFree(device_sums));

    std::ofstream result(argv[6], std::ios::binary);
    result.write(reinterpret_cast<const char *>(output.data()), count);
    std::ofstream stats(argv[7]);
    stats.precision(17);
    stats << "{\"sum\":" << total << ",\"mean\":" << mean << ",\"gain\":" << gain
          << ",\"blocks\":" << blocks << ",\"threads\":" << threads << "}\n";
    std::ofstream partials(argv[8]);
    for (auto sum : sums) partials << sum << '\n';
    result.close(); stats.close(); partials.close();
    if (!result || !stats || !partials) {
        std::fprintf(stderr, "Failed to write output files.\n");
        return EXIT_FAILURE;
    }
    std::printf("GPU: %s\nBlocks: %d, threads per block: %d\n", device.name, blocks, threads);
    std::printf("Input mean: %.3f, target: %d, gain: %.3f\n", mean, target, gain);
    if (mean == 0.0) std::printf("All pixels are black; multiplicative adjustment leaves them black.\n");
    return EXIT_SUCCESS;
}
