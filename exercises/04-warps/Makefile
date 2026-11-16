NVCC ?= nvcc
NVCC_FLAGS ?= -O2

.PHONY: all clean
all: warp_timing

warp_timing: warp_timing.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f warp_timing
