# Makefile for CUDA code with dynamic parallelism

NVCC = nvcc
NVCC_FLAGS = -O2 -rdc=true
LDFLAGS = -lcudadevrt

# List of source files
SRCS = vector_add_async.cu compare_graphs.cu custom_graph.cu update_node_args.cu

# Generate target executables by stripping .cu extension
TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

# Compile each .cu file with support for dynamic parallelism
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
