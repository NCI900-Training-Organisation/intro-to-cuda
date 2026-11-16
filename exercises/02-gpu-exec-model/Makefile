NVCC ?= nvcc
NVCC_FLAGS ?= -O2

.PHONY: all clean
all: thread_indexing print_launch print_global_index double_thread_index double_global_index

thread_indexing: thread_indexing.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

print_launch: print_launch.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

print_global_index: print_global_index.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

double_thread_index: double_thread_index.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

double_global_index: double_global_index.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f thread_indexing print_launch print_global_index double_thread_index double_global_index
