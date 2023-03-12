#include <stdio.h>
#include "cuda_helpers.h"

/*
 axpy kernel function
*/
__global__ void axpy(const float a, const float* X, const float* Y, float* Z, const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) Z[idx] = a * X[idx] + Y[idx];
}



/*
  Run the axpy function:
    - Copy X and Y to the device
    - Run the kernel
    - Copy Z back to the host
  TODO: MODIFY THIS TO USE STREAMS FOR OVERLAPPED COMPUTE AND DATA TRANSFER
        Consider taking an array of streams and the stream size as extra input
        arguments to avoid creating/destroying the streams every time.
*/
void run_axpy(
  const float a, 
  float* X, float* X_d,
  const float* Y, float* Y_d,
  float* Z, float* Z_d,
  const int N
) {
  // Copy X and Y to the GPU
  cuda_check(cudaMemcpy(X_d, X, sizeof(*X)*N, cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(Y_d, Y, sizeof(*Y)*N, cudaMemcpyHostToDevice));

  // Call the axpy kernel with at least N threads
  axpy<<<DEF_GRID(N), DEF_BLOCK>>>(a, X_d, Y_d, Z_d, N);
  cuda_check(cudaGetLastError());

  // Copy the result back into Z
  cuda_check(cudaMemcpy(Z, Z_d, sizeof(*Z)*N, cudaMemcpyDeviceToHost));
}




/*
  Test the axpy function to make sure it gives correct results
*/
typedef enum {SUCCESS=0, FAIL} TestResult;

TestResult test_axpy(const int N) {
  // Allocate memory
  // TODO: CHANGE THIS TO USE cudaHostMalloc() FOR PINNED MEMORY
  float* X = (float*)malloc(sizeof(*X) * N);
  float* Y = (float*)malloc(sizeof(*Y) * N);
  float* Z = (float*)malloc(sizeof(*Z) * N);
  
  // Set up initial values of a, X and Y
  float a = 2.0;
  for (int i = 0; i < N; ++i) {
    X[i] = i;
    Y[i] = 2.*(N-i-1);
  }
  
  // Allocate memory on the GPU for X, Y, Z
  float* X_d; cuda_check(cudaMalloc((void**)&X_d, sizeof(*X_d) * N));
  float* Y_d; cuda_check(cudaMalloc((void**)&Y_d, sizeof(*Y_d) * N));
  float* Z_d; cuda_check(cudaMalloc((void**)&Z_d, sizeof(*Z_d) * N));
  
  
  // TODO: SET UP STREAMS
  
  // Run the kernel 100 times for better statistics
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  const int N_TESTS = 1000;
  double millis_total;
  for (int test = 0; test < N_TESTS; ++test) {
    /// Begin timing
    cuda_check(cudaDeviceSynchronize());
    cudaEventRecord(start);
    
    // Run the kernel
    run_axpy(a, X, X_d, Y, Y_d, Z, Z_d, N);
    
    /// End timing
    cuda_check(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecs = 0;
    cudaEventElapsedTime(&millisecs, start, stop);
    millis_total += millisecs;
  }
  // Calculate average execution time
  printf("%10g ms  ...  ", millis_total / N_TESTS);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // Check the results are correct
  TestResult status = SUCCESS;
  for (int i = 0; i < N; ++i) {
    if (Z[i] != a*X[i] + Y[i]) {
      status = FAIL;
      break;
    }
  }
  
  // Clean up memory
  // TODO: USE cudaFreeHost() FOR PINNED MEMORY
  free(X);
  free(Y);
  free(Z);
  cudaFree(X_d);
  cudaFree(Y_d);
  cudaFree(Z_d);
  
  // TODO: CLEAN UP STREAMS
  
  return status;
}



int main(void) {
  int TESTS[] = {1024, 10000, 500000, 5000000, 10000000};
  for (int i = 0; i < sizeof(TESTS)/sizeof(*TESTS); ++i) {
    printf("Testing axpy for N = %-10i...  ", TESTS[i]);
    fflush(stdout);
    if (test_axpy(TESTS[i]) == SUCCESS)
      printf("Passed!\n");
    else {
      printf("Failed!\n");
      return 1;
    }
  }
  printf("\nAll tests passed!\n");
  return 0;
}
