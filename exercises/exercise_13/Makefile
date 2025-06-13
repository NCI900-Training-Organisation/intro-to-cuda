# Makefile for multiple CUDA source files with Nsight Systems profiling

NVCC = nvcc

# Enable debugging info if needed (e.g., make ENABLE_DEBUG=1)

NVCC_FLAGS = -G -g 


# List of CUDA source files
SRCS = 18_vector_add.cu

# Generate target executable names by stripping .cu extension
TARGETS = $(SRCS:.cu=)

# Default target
all: $(TARGETS)

# Compile each .cu file into its corresponding executable
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<



# Clean build and profiling output files
clean:
	rm -f $(TARGETS) 
