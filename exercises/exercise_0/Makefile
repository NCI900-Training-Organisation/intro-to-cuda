# -------------------------------
# Compilers
# -------------------------------
NVCC = nvcc
CC   = gcc

# -------------------------------
# Compiler flags
# -------------------------------
NVCC_FLAGS = -O2
C_FLAGS    = -O2

# -------------------------------
# CUDA libraries
# -------------------------------
CUDA_LIBS = -lcublas

# -------------------------------
# Intel MKL libraries (no hardcoded paths)
# Assume MKL environment has been sourced:
# source /apps/intel-tools/intel-mkl/2024.2.1/env/vars.sh intel64
# -------------------------------
MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# -------------------------------
# Source files
# -------------------------------
CUDA_SRCS = 0_perf_gemm_cuda.cu
MKL_SRCS  = 0_perf_gemm_intelmkl.c

# -------------------------------
# Targets
# -------------------------------
CUDA_TARGETS = $(CUDA_SRCS:.cu=)
MKL_TARGETS  = $(MKL_SRCS:.c=)
TARGETS = $(CUDA_TARGETS) $(MKL_TARGETS)

# -------------------------------
# Default target
# -------------------------------
all: $(TARGETS)

# -------------------------------
# Rule for CUDA targets
# -------------------------------
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)

# -------------------------------
# Rule for MKL targets
# -------------------------------
%: %.c
	$(CC) $(C_FLAGS) -o $@ $< $(MKL_LIBS)

# -------------------------------
# Clean target
# -------------------------------
clean:
	rm -f $(TARGETS)
