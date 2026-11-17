#pragma once
#include <cstdio>

// Host-side output: call after the result copy/prefetch has completed,
// outside the timed region and before freeing any input or output buffers.
inline void print_vector_add_output(const float *a, const float *b, const float *c,
                                     int count, int iteration = -1)
{
    const int shown = count < 8 ? count : 8;
    std::printf("Vector-add output: first %d of %d elements", shown, count);
    if (iteration >= 0) std::printf(" (iteration %d)", iteration);
    std::printf("\nIndex        A        B    C = A + B\n");
    for (int i = 0; i < shown; ++i)
        std::printf("%5d %8.1f %8.1f %12.1f\n", i, a[i], b[i], c[i]);
}
