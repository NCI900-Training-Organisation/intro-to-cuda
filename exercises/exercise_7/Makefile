# Makefile for CUDA code with dynamic parallelism

NVCC = nvcc
NVCC_FLAGS = -O2 -rdc=true
LDFLAGS = -lcudadevrt

# List of source files
SRCS = 8_vector_add_async.cu

# Generate target executables by stripping .cu extension
TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

# Compile each .cu file with support for dynamic parallelism
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
