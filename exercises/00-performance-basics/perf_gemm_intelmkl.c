#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

int main(void)
{
    int N = 1000;       // Matrix dimension
    int runs = 5;       // Number of GEMM runs

    // Allocate matrices on CPU
    double *A = (double *)malloc(sizeof(double) * N * N);
    double *B = (double *)malloc(sizeof(double) * N * N);
    double *C = (double *)malloc(sizeof(double) * N * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    double alpha = 1.0;
    double beta  = 1.0;

    // Time each run using clock_gettime
    struct timespec start, end;

    for (int i = 0; i < runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Perform GEMM: C = alpha * A * B + beta * C
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                            (end.tv_nsec - start.tv_nsec) / 1e6;

        printf("Run %d completed in %.3f ms\n", i + 1, elapsed_ms);
    }

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
