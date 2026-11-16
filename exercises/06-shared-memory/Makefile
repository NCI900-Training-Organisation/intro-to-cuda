NVCC ?= nvcc
NVCC_FLAGS ?= -O2

.PHONY: all clean
all: brightness

brightness: brightness.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f brightness
