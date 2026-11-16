NVCC ?= nvcc
NVCC_FLAGS ?= -O2

.PHONY: all clean
all: thread_indexing

thread_indexing: thread_indexing.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f thread_indexing
