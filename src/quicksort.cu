#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Swap utility
__device__ void swap(int *data, int i, int j) {
    int tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}

// Kernel to do partition in parallel
__device__ int parallelPartition(int *data, int left, int right, int *pivotIndex) {
    __shared__ int ltCount[BLOCK_SIZE];  // Count of elements less than pivot
    __shared__ int gtCount[BLOCK_SIZE];  // Count of elements greater than pivot

    int tid = threadIdx.x;
    int pivot = data[right];
    int val = 0;

    if (left + tid <= right) {
        val = data[left + tid];
        ltCount[tid] = (val < pivot) ? 1 : 0;
        gtCount[tid] = (val > pivot) ? 1 : 0;
    } else {
        ltCount[tid] = 0;
        gtCount[tid] = 0;
    }

    __syncthreads();

    // Prefix sum for less-than counts
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int temp = 0;
        if (tid >= stride)
            temp = ltCount[tid - stride];
        __syncthreads();
        ltCount[tid] += temp;
        __syncthreads();
    }

    int totalLt = ltCount[BLOCK_SIZE - 1];
    int writeIndex = left + ltCount[tid] - 1;

    if (left + tid <= right && val < pivot) {
        data[writeIndex] = val;
    }

    __syncthreads();

    // Move pivot to its correct position
    if (tid == 0) {
        int pivotPos = left + ltCount[BLOCK_SIZE - 1];
        swap(data, pivotPos, right);
        *pivotIndex = pivotPos;
    }

    return 0;
}

// Recursive quicksort kernel
__global__ void quickSortKernel(int *data, int left, int right) {
    if (left < right) {
        int pivotIndex;
        parallelPartition(data, left, right, &pivotIndex);
        __syncthreads();

        if (pivotIndex > left)
            quickSortKernel<<<1, BLOCK_SIZE>>>(data, left, pivotIndex - 1);
        cudaDeviceSynchronize();

        if (pivotIndex < right)
            quickSortKernel<<<1, BLOCK_SIZE>>>(data, pivotIndex + 1, right);
        cudaDeviceSynchronize();
    }
}

int main() {
    const int N = 1024;
    int h_data[N];

    // Initialize array in reverse order
    for (int i = 0; i < N; ++i)
        h_data[i] = N - i;

    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    quickSortKernel<<<1, BLOCK_SIZE>>>(d_data, 0, N - 1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array:\n");
    for (int i = 0; i < 20; ++i)
        printf("%d ", h_data[i]);
    printf("...\n");

    cudaFree(d_data);
    return 0;
}



