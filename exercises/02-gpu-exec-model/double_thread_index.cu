// Intentionally incorrect: both blocks use the same local indices, causing a data race.
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

__global__ void double_values(int *values, int count)
{
    int i = threadIdx.x;
    if (i < count) {
        values[i] *= 2;
    }
}

int main()
{
    constexpr int count = 8;
    const int input[count] = {1, 2, 3, 4, 5, 6, 7, 8};
    int output[count];
    int *device_values = nullptr;

    std::printf("Input:   ");
    for (int value : input) std::printf("%d ", value);
    std::puts("");

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_values), sizeof(input)));
    CUDA_CHECK(cudaMemcpy(device_values, input, sizeof(input), cudaMemcpyHostToDevice));

    // Two blocks, four threads per block: eight threads for eight elements.
    double_values<<<2, 4>>>(device_values, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, device_values, sizeof(output), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_values));

    std::printf("Output:  ");
    for (int value : output) std::printf("%d ", value);
    std::puts("\nExpected: 2 4 6 8 10 12 14 16");
    std::puts("Intentional bug: both blocks address elements 0-3; elements 4-7 are never addressed.");
    std::puts("The overlapping read/modify/write operations race: do not rely on the first four results.");
    return EXIT_SUCCESS;
}
