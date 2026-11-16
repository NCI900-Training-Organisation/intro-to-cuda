// Correct: use a unique global index for each array element.
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

__global__ void double_values(int *values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    values[i] *= 2;

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
    double_values<<<2, 4>>>(device_values);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, device_values, sizeof(output), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_values));

    std::printf("Output:  ");
    for (int value : output) std::printf("%d ", value);
    std::puts("\nExpected: 2 4 6 8 10 12 14 16");
    for (int i = 0; i < count; ++i) {
        if (output[i] != 2 * input[i]) {
            std::fprintf(stderr, "FAIL: element %d was not doubled correctly.\n", i);
            return EXIT_FAILURE;
        }
    }
    std::puts("PASS: every element was doubled exactly once.");
    return EXIT_SUCCESS;
}
