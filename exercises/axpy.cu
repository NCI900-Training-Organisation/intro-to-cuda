#include <stdio.h>
#include "cuda_helpers.h"

/*
 TODO: Add your axpy kernel function here
*/


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
  
  // TODO: Copy X and Y to the GPU
  
  // TODO: Call the axpy kernel with at least N threads
  
  // TODO: Copy the result back into Z
  
  
  // Check the results are correct
  TestResult status = SUCCESS;
  for (int i = 0; i < N; ++i) {
    if (Z[i] != a*X[i] + Y[i]) {
      status = FAIL;
      break;
    }
  }
  
  // Clean up memory
  free(X);
  free(Y);
  free(Z);
  
  // TODO: Clean up GPU memory
  
  return status;
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
