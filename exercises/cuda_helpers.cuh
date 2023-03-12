#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

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

#endif
