# Makefile for multiple CUDA source files

NVCC = nvcc
NVCC_FLAGS = -O2

# List of source files
SRCS = 2_vector_add.cu

# Generate target executables by stripping .cu extension
TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

# Compile each .cu file to its corresponding executable
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

