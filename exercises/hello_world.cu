#include <stdio.h>

__global__
void hello() {
  printf("Hello, world!\n");
}

int main(void) {
  hello<<<1,64>>>();
  cudaDeviceSynchronize();
  return 0;
}