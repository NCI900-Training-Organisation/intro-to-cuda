# Makefile for CUDA code with dynamic parallelism and cuBLAS

NVCC      := nvcc
NVCC_FLAGS:= -O2 -rdc=true
LDFLAGS   := -lcudadevrt -lcublas

# List of source files
SRCS      := 16_multi-GPU.cu

# Generate target executables by stripping .cu extension
TARGETS   := $(SRCS:.cu=)

all: $(TARGETS)

# Compile each .cu file with support for dynamic parallelism (and cuBLAS)
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
