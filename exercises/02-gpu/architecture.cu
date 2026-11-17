#include <stdio.h>
#include <cuda_runtime.h>

// Helper function to get CUDA cores per SM based on compute capability
int getCoresPerSM(int major, int minor) 
{
    typedef struct {
        int major;
        int minor;
        int cores;
    } SMVersion;

    SMVersion coreTable[] = {
        {2, 0, 32},  // Fermi
        {2, 1, 48},
        {3, 0, 192}, // Kepler
        {3, 5, 192},
        {3, 7, 192},
        {5, 0, 128}, // Maxwell
        {5, 2, 128},
        {5, 3, 128},
        {6, 0, 64},  // Pascal
        {6, 1, 128},
        {6, 2, 128},
        {7, 0, 64},  // Volta
        {7, 2, 64},  // Xavier
        {7, 5, 64},  // Turing
        {8, 0, 64},  // Ampere GA100
        {8, 6, 128}, // Ampere GA10x
        {8, 9, 128}, // Ada Lovelace (tentative)
        {9, 0, 128}, // Future (hypothetical)
        {-1, -1, -1}
    };

    int i = 0;
    while (coreTable[i].major != -1) 
    {
        if (coreTable[i].major == major && coreTable[i].minor == minor) {
            return coreTable[i].cores;
        }
        i++;
    }
    printf("Unknown SM version %d.%d - defaulting cores/SM=64\n", major, minor);
    return 64; // fallback default
}

int main(void) 
{

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0); // Query device 0
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device Model: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    int coresPerSM = getCoresPerSM(prop.major, prop.minor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("CUDA Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", coresPerSM * prop.multiProcessorCount);

    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    // Estimate max blocks per SM (rough)
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    printf("Estimated max blocks per SM: %d\n", maxBlocksPerSM);

    // Tensor cores available if compute capability >= 7.0
    printf("Tensor cores available: %s\n", (prop.major >= 7) ? "Yes" : "No");


    // L1 cache size is not directly given, but sharedMemPerBlock is
    printf("Shared memory per block (approx. L1 cache): %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    printf("L2 Cache size: %.2f KB\n", prop.l2CacheSize / 1024.0f);

    return 0;
}
