// Record a 1D or 2D launch. All dimensions are passed explicitly on the command line.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct ThreadRecord { int global, local, bx, by, tx, ty, x, y; };

__global__ void record_indices(ThreadRecord *records)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int global = y * width + x;
    int local = threadIdx.y * blockDim.x + threadIdx.x;
    records[global] = {global, local, int(blockIdx.x), int(blockIdx.y),
                       int(threadIdx.x), int(threadIdx.y), x, y};
}

int dimension(const char *text)
{
    char *end;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || *end || end == text || value < 1 || value > 16) {
        std::fprintf(stderr, "Each dimension must be an integer from 1 to 16.\n");
        std::exit(EXIT_FAILURE);
    }
    return int(value);
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        std::fprintf(stderr, "Usage: %s grid_x grid_y block_x block_y output.csv\n", argv[0]);
        return EXIT_FAILURE;
    }
    dim3 grid(dimension(argv[1]), dimension(argv[2]));
    dim3 block(dimension(argv[3]), dimension(argv[4]));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (block.x * block.y > unsigned(prop.maxThreadsPerBlock)) {
        std::fprintf(stderr, "Too many threads per block for this GPU.\n");
        return EXIT_FAILURE;
    }
    int count = grid.x * grid.y * block.x * block.y;
    ThreadRecord *device_records = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_records), count * sizeof(ThreadRecord)));
    record_indices<<<grid, block>>>(device_records);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ThreadRecord *records = new ThreadRecord[count];
    CUDA_CHECK(cudaMemcpy(records, device_records, count * sizeof(ThreadRecord), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_records));
    FILE *file = std::fopen(argv[5], "w");
    if (!file) { std::perror(argv[5]); delete[] records; return EXIT_FAILURE; }
    std::fprintf(file, "global,local,bx,by,tx,ty,x,y\n");
    for (int i = 0; i < count; ++i) {
        const auto &r = records[i];
        std::fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%d\n", r.global, r.local,
                     r.bx, r.by, r.tx, r.ty, r.x, r.y);
    }
    delete[] records;
    if (std::fclose(file)) { std::perror(argv[5]); return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}
