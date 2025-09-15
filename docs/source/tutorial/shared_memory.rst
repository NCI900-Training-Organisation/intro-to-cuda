Shared Memory in CUDA
========================

.. admonition:: Overview
   :class: Overview

    * **Time:** 30 min

    #. Learn about shared memory in CUDA.
    #. Understand the characteristics and usage of shared memory.


Shared memory is a type of memory that is shared among threads within the same block in CUDA. It is much 
faster than global memory and can be used to store data that needs to be accessed by multiple threads. 
Shared memory is particularly useful for reducing global memory accesses and improving performance in 
parallel algorithms.

Shared Memory Characteristics
----------------------------
Shared memory in CUDA has several key characteristics:

1. **Scope**: Shared memory is accessible only to threads within the same block. Threads in different blocks cannot access each other's shared memory.

2. **Speed**: Shared memory is on-chip memory, making it much faster than global memory (DRAM). Accessing shared memory typically has a latency of around 10-50 cycles, compared to 400-800 cycles for global memory.

3. **Size**: The size of shared memory is limited, typically ranging from 48 KB to 96 KB per block, depending on the GPU architecture.

4. **Coherency**: Shared memory is coherent across all threads in a block, meaning that changes made by one thread are visible to others in the same block.

5. **Usage**: Shared memory is often used for data that needs to be accessed frequently by multiple threads, such as intermediate results in parallel computations.

.. list-table:: Shared Memory vs DRAM in NVIDIA GPUs
    - Very fast (10–50 cycles)
    - Slow (400–800 cycles)
    - 
    - Small (48–96 KB per SM)
    - Large (GBs)
    - 
    - Used for frequently accessed data within a block
    - Used for data shared across all blocks
    - 
    - Reduces global memory traffic and accelerates access
    - Higher latency, but larger capacity

There are two main types of shared memory in CUDA:

1. **Static Shared Memory**: This is defined at compile time and has a fixed size. It is declared using the `__shared__` keyword in the kernel code. Static shared memory is allocated for each block and is initialized when the block is launched.

.. code-block:: c
    :linenos:

     __global__ void addKernel(int *input, int *output) 
    {
        __shared__ int sdata[256];  // dynamic shared memory

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Load data from global to shared memory
        sdata[tid] = input[idx];
        __syncthreads();

        // Do something with shared memory
        sdata[tid] += 10;
        __syncthreads();

        // Store result back to global memory
        output[idx] = sdata[tid];
    }

2. **Dynamic Shared Memory**: This is allocated at runtime and can vary in size based on the needs of the kernel. It is declared using the `extern __shared__` keyword in the kernel code. Dynamic shared memory allows for more flexibility, as its size can be specified when launching the kernel.

.. code-block:: c
    :linenos:

    
    __global__ void addKernel(int *input, int *output) 
    {
        extern __shared__ int sdata[];  // dynamic shared memory
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Load data from global to shared memory
        sdata[tid] = input[idx];
        __syncthreads();

        // Do something with shared memory
        sdata[tid] += 10;
        __syncthreads();

        // Store result back to global memory
        output[idx] = sdata[tid];
    }

    int main() 
    {
        int threads = 256;
        int sharedMemSize = threads * sizeof(int);  // Size of dynamic shared memory

        ...
        ...
        ...

        addKernel<<<1, threads, sharedMemSize>>>(d_input, d_output);
    }



.. important::

    You declare only one ``extern __shared__ array`` inside your kernel.

.. admonition:: Explanation
   :class: attention

    ``__syncthreads()`` is a barrier synchronization function in CUDA that ensures all threads in the same 
    thread block:

    1. Reach the barrier, and
    2. Complete all memory accesses (reads/writes to shared memory)

    before any thread in the block continues past it.

.. admonition:: Key Points
   :class: hint
   
    #. Shared memory is a fast, on-chip memory accessible by threads within the same block.
    #. It is used to reduce global memory accesses and improve performance in parallel algorithms.
    #. There are two types of shared memory: static (fixed size) and dynamic (runtime size).
    #. The `__syncthreads()` function is used to synchronize threads within a block, ensuring all threads reach a certain point before proceeding.