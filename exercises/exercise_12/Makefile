# Makefile for multiple CUDA source files with Nsight Systems profiling

NVCC = nvcc

# Enable debugging info if needed (e.g., make ENABLE_DEBUG=1)
ifdef ENABLE_DEBUG
	NVCC_FLAGS = -G -lineinfo
else
	NVCC_FLAGS = -O2 -lineinfo
endif

# List of CUDA source files
SRCS = 17_vector_add_unified.cu

# Generate target executable names by stripping .cu extension
TARGETS = $(SRCS:.cu=)

# Default target
all: $(TARGETS)

# Compile each .cu file into its corresponding executable
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run Nsight Systems profiler on all targets
profile: $(TARGETS)
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile \
			--stats=true \
			--trace=cuda,nvtx,osrt \
			--cuda-memory-usage=true \
			-o $${exe}_profile \
			./$${exe}; \
	done

# Clean build and profiling output files
clean:
	rm -f $(TARGETS) *.qdrep *.sqlite *.nsys-rep
