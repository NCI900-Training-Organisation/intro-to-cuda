#include <stdio.h>

/*
  Helper functions for error checking
*/
#define cuda_check_impl(cmd, abort) {                  \
  cudaError_t status = (cmd);                          \
  if (status != cudaSuccess) {                         \
    fprintf(stderr, "CUDA Error: %s (%s:%d)\n",        \
      cudaGetErrorString(status), __FILE__, __LINE__); \
    if (abort) exit(status);                           \
  }                                                    \
}
#define cuda_check(cmd) cuda_check_impl((cmd), 1)
#define cuda_check_noabort(cmd) cuda_check_impl((cmd), 0)


/*
  Helper functions for calculating grid size
*/
#define NBLOCKS(N, BLOCK_SIZE) (((N) + (BLOCK_SIZE) - 1)/(BLOCK_SIZE))
#define DEF_BLOCK 256
#define DEF_GRID(N) NBLOCKS((N), DEF_BLOCK)


/*
 TODO: Add your axpy function here
*/
__global__ void axpy(const float a, const float* X, const float* Y, float* Z, const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) Z[idx] = a * X[idx] + Y[idx];
}


/*
  Test the axpy function to make sure it gives correct results
*/
typedef enum {SUCCESS=0, FAIL} TestResult;

TestResult test_axpy(const int N) {
  // Allocate memory
  float* X = (float*)malloc(sizeof(*X) * N);
  float* Y = (float*)malloc(sizeof(*Y) * N);
  float* Z = (float*)malloc(sizeof(*Z) * N);
  
  // Set up initial values of a, X and Y
  float a = 2.0;
  for (int i = 0; i < N; ++i) {
    X[i] = i;
    Y[i] = 2.*(N-i-1);
  }
  
  // TODO: allocate memory on the GPU for X, Y, Z
  float* X_d; cuda_check(cudaMalloc((void**)&X_d, sizeof(*X_d) * N));
  float* Y_d; cuda_check(cudaMalloc((void**)&Y_d, sizeof(*Y_d) * N));
  float* Z_d; cuda_check(cudaMalloc((void**)&Z_d, sizeof(*Z_d) * N));
  
  // TODO: Copy X and Y to the GPU
  cuda_check(cudaMemcpy(X_d, X, sizeof(*X)*N, cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(Y_d, Y, sizeof(*Y)*N, cudaMemcpyHostToDevice));
  
  // TODO: Call the axpy kernel with at least N threads
  axpy<<<DEF_GRID(N), DEF_BLOCK>>>(a, X_d, Y_d, Z_d, N);
  cuda_check(cudaGetLastError());
  
  // TODO: Copy the result back into Z
  cuda_check(cudaMemcpy(Z, Z_d, sizeof(*Z)*N, cudaMemcpyDeviceToHost));
  
  // Check the results are correct
  for (int i = 0; i < N; ++i) {
    if (Z[i] != a*X[i] + Y[i]) return FAIL;
  }
  return SUCCESS;
}


int main(void) {
  int TESTS[] = {1024, 10000, 500000};
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